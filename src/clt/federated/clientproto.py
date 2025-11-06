import copy
import torch
import torch.nn as nn
import numpy as np
import time
from collections import defaultdict
import torchvision.transforms as transforms
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from models.mae import MaskedAutoencoderViT
from models.classifier import *
from data.dataset_utils import *
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from collections import Counter

class clientProto(object):
    def __init__(self, cfg, id, dataset, img_path, csv_path, **kwargs):

        self.protos = None
        self.global_protos = None
        self.loss_mse = nn.MSELoss()
        self.loss = nn.CrossEntropyLoss()

        self.dataset = dataset
        self.lamda = float(cfg['COMMON']['LDA'])
        self.device = cfg['COMMON']['DEVICE']
        self.img_size = cfg['COMMON']['IMG_SIZE']
        self.patch_size = cfg['COMMON']['PATCH_SIZE']
        self.mean = cfg['COMMON']['MEAN']
        self.std = cfg['COMMON']['STD']
        self.local_epochs = cfg['COMMON']['LOCAL_EPOCH']
        self.batch_size = cfg['COMMON']['BATCH_SIZE']
        self.weight_decay = float(cfg['COMMON']['WEIGHT_DECAY'])
        self.train_ratio = cfg['COMMON']['TRAIN_RATIO']
        self.lr = float(cfg['COMMON']['LEARNING_RATE'])
        self.adapft = cfg['COMMON']['ADAPFT']
        self.num_classes = cfg['COMMON']['NUM_CLASSES']
        self.model_path = cfg['COMMON']['MODEL_PATH']
        self.momentum = float(cfg['COMMON']['MOMENTUM'])
        self.partial_label_ratio = cfg['COMMON']['PARTIAL_LABEL_RATIO']
        self.partial_seed = cfg['COMMON']['PARTIAL_SEED']

        self.dataset_name = cfg['CLIENTS'][self.dataset]['NAME']

        self.init_model()

        self.optimizer = torch.optim.SGD(self.update_params, lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=cfg['SERVER']['GLOBAL_ROUNDS'], eta_min=5e-5)


        self.scaler = GradScaler()

        self.img_path = img_path
        self.csv_path = csv_path

        self.train_transform = transforms.Compose([
                    transforms.RandomResizedCrop(self.img_size, scale=(0.5, 1.0)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(degrees=5),
                    transforms.ToTensor(),
                    transforms.Normalize(cfg['COMMON']['MEAN'], cfg['COMMON']['STD'])])
        

        self.transform = transforms.Compose([
                    transforms.Resize(size=self.img_size),
                    transforms.CenterCrop(size=(self.img_size, self.img_size)), 
                    transforms.ToTensor(),
                    transforms.Normalize(cfg['COMMON']['MEAN'], cfg['COMMON']['STD'])])

        self.id = id

        self.train_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
        self.send_time_cost = {'num_rounds': 0, 'total_cost': 0.0}

    def init_model(self):
        pretrained_mae = MaskedAutoencoderViT(
            img_size=self.img_size,
            patch_size=self.patch_size,
            hybrid=False
        ).to(self.device)

        print("Use Pretrained")
        state_dict = torch.load(self.model_path, map_location=self.device, weights_only=True)

        if isinstance(state_dict, nn.Module):
            state_dict = state_dict.state_dict()
            print("LOAD")

        pretrained_mae.load_state_dict(state_dict, strict=True)
        pretrained_mae.eval()

        if self.adapft:
            self.model = FineTunedMAE_Shallow(pretrained_mae, num_classes=self.num_classes, freeze=True).to(self.device)
            self.update_params = list(self.model.adapter.parameters()) + \
              list(self.model.head.parameters())
            print("Adapter Fine-tuning")
        else:
            self.model = FineTunedMAE(pretrained_mae, num_classes=self.num_classes, freeze=False).to(self.device)
            self.update_params = list(self.model.parameters())
            print("No Adapter, Full Fine-tuning")

    def load_train_data(self):
        if hasattr(self, '_trainloader'):
            return self._trainloader

        full_dataset = eval(self.dataset_name)(
            self.img_path,
            self.csv_path,
            train_ratio=self.train_ratio,
            train=True,
            transform=self.train_transform
        )

        if self.partial_label_ratio < 1.0 and self.partial_seed is not None:
            dataset = get_partial_labeled_subset(
                full_dataset,
                label_ratio=self.partial_label_ratio,
                random_state=self.partial_seed
            )
        else:
            dataset = full_dataset

        if self.train_time_cost['num_rounds'] == 0:
            print("\n📦 [CLIENT DATA LOADING DEBUG]")
            print(f"🧩 Client ID: {self.id} | Dataset: {self.dataset_name}")
            print(f"🔢 Total samples in full dataset: {len(full_dataset)}")
            print(f"🎯 PARTIAL_LABEL_RATIO: {self.partial_label_ratio} | PARTIAL_SEED: {self.partial_seed}")
            if self.partial_label_ratio < 1.0 and self.partial_seed is not None:
                print(f"✅ Partial label subset selected: {len(dataset)} samples")
            else:
                print("✅ Full label dataset used (no partial sampling)")

            label_counter = Counter()
            for _, label in dataset:
                label_value = int(label.item()) if isinstance(label, torch.Tensor) else int(label)
                label_counter[label_value] += 1
            print(f"📊 Label distribution in training set: {dict(label_counter)}\n")

        self._trainloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
        return self._trainloader
    
    def load_test_data(self):
        dataset = eval(self.dataset_name)(self.img_path, self.csv_path, train_ratio=self.train_ratio, train=False, transform=self.transform)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

    def train(self):
        trainloader = self.load_train_data()
        self.model.train()
        proto_buf = defaultdict(list)
        start_time = time.time()

        for _ in range(self.local_epochs):
            for x, y in trainloader:
                x, y = x.to(self.device), y.long().to(self.device)
                self.optimizer.zero_grad(set_to_none=True)

                with autocast():
                    rep   = self.model.extract_representation(x)  # (B,D)
                    logit = self.model.head(rep)
                    loss_ce = self.loss(logit, y)

                    if self.global_protos is not None:
                        proto_tgt = torch.stack([self.global_protos[c] for c in y.tolist()]).to(self.device)
                        pos_term = self.loss_mse(rep, proto_tgt)
                    else:
                        pos_term = torch.tensor(0.0, device=self.device)

                    loss_sup = loss_ce + self.lamda * pos_term


                # backward & step
                self.scaler.scale(loss_sup).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.update_params, 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                for idx, cls in enumerate(y):
                    proto_buf[cls.item()].append(rep[idx].detach())

        self.protos = agg_func(proto_buf)
        self.scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time
# ─────────────────────────────────────────────────────────────


    def set_protos(self, global_protos):
        self.global_protos = global_protos

    def collect_protos(self):
        trainloader = self.load_train_data()
        self.model.eval()

        protos = defaultdict(list)
        with torch.no_grad():
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                rep = self.model.extract_representation(x)

                for i, yy in enumerate(y):
                    y_c = yy.item()
                    protos[y_c].append(rep[i, :].detach().data)

        self.protos = agg_func(protos)

    def collect_protos_with_stats(self):
        trainloader = self.load_train_data()
        self.model.eval()

        protos = defaultdict(list)
        with torch.no_grad():
            for x, y in trainloader:
                x, y = x.to(self.device), y.to(self.device)
                rep = self.model.extract_representation(x)

                for i, cls in enumerate(y):
                    protos[int(cls.item())].append(rep[i])

        proto_stats = {}
        for cls, reps in protos.items():
            reps = torch.stack(reps)  # (n, D)
            mean = reps.mean(dim=0)
            var  = reps.var(dim=0, unbiased=False).mean() 
            count = reps.size(0)
            proto_stats[cls] = {"proto": mean, "var": var, "count": count}

        return proto_stats

    def _calc_distance(self, rep_vec, proto_vec):
        return (rep_vec - proto_vec).pow(2).sum(dim=-1)  # (B,)

    def test_metrics(self):
        testloader = self.load_test_data()
        self.model.eval()

        correct, total = 0, 0
        all_preds = []
        all_labels = []

        if self.global_protos is None:
            return 0, 1e-5, 0

        with torch.no_grad():
            for x, y in testloader:
                x, y = x.to(self.device), y.long().to(self.device)
                rep = self.model.extract_representation(x)  # (B, D)

                dist_mat = torch.full((y.size(0), self.num_classes), float('inf'), device=self.device)

                for j, gp in self.global_protos.items():
                    if isinstance(gp, list):
                        continue  
                    dist_mat[:, j] = self._calc_distance(rep, gp)

                preds = torch.argmin(dist_mat, dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(y.cpu().tolist())

        cm = confusion_matrix(all_labels, all_preds).tolist()
        print(f'{self.dataset_name}: {cm}') 

        return correct, total, 0

    def train_metrics(self):
        trainloader = self.load_train_data()
        self.model.eval()

        total, agg_loss = 0, 0.0
        with torch.no_grad():
            for x, y in trainloader:
                x, y = x.to(self.device), y.long().to(self.device)
                
                with autocast():
                    rep   = self.model.extract_representation(x)
                    logit = self.model.head(rep)

                    loss_batch = self.loss(logit, y)

                    if self.global_protos is not None:                   
                        proto_tgt = torch.stack(
                            [self.global_protos[c] for c in y.tolist()]
                        ).to(self.device)
                        pos_term = self.loss_mse(rep, proto_tgt)

                        loss_batch += self.lamda * pos_term     

                agg_loss += loss_batch.item() * y.size(0)
                total    += y.size(0)

        return agg_loss, total


    def save_model(self, rnd, save_dir=None):
        if save_dir is None:
            raise ValueError("save_dir must be specified.")

        os.makedirs(save_dir, exist_ok=True)

        file_name = f"client_{self.dataset_name}_model_{rnd}.pt"
        save_path = os.path.join(save_dir, file_name)

        if self.adapft:
            state_dict = {
                'adapter': self.model.adapter.state_dict(),
                'head': self.model.head.state_dict()
            }
        else:
            state_dict = self.model.state_dict()

        torch.save(state_dict, save_path)
        print(f"✅ Model saved to: {save_path}")



# https://github.com/yuetan031/fedproto/blob/main/lib/utils.py#L205
def agg_func(protos):
    """
    Returns the average of the weights.
    """

    for [label, proto_list] in protos.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            protos[label] = proto / len(proto_list)
        else:
            protos[label] = proto_list[0]

    return protos