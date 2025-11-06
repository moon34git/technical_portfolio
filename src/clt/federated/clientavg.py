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

class clientAvg(object):
    def __init__(self, cfg, id, dataset, img_path, csv_path, **kwargs):

        self.loss = nn.CrossEntropyLoss()

        self.dataset = dataset
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
        self.dataset_name = cfg['CLIENTS'][self.dataset]['NAME']
        self.momentum = float(cfg['COMMON']['MOMENTUM'])
        self.partial_label_ratio = cfg['COMMON']['PARTIAL_LABEL_RATIO']
        self.partial_seed = cfg['COMMON']['PARTIAL_SEED']

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
        start_time = time.time()


        for _ in range(self.local_epochs):
            for x, y in trainloader:
                x, y = x.to(self.device), y.long().to(self.device)
                self.optimizer.zero_grad(set_to_none=True)

                with autocast():
                    rep   = self.model.extract_representation(x)  # (B,D)
                    logit = self.model.head(rep)
                    loss = self.loss(logit, y)

                # backward & step
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.update_params, 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()

        self.scheduler.step()
        self.train_samples = len(trainloader.dataset) 

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time
# ─────────────────────────────────────────────────────────────
        
    def set_parameters(self, model):
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()

    def set_adapter_head(self, adapter_state, head_state):
        self.model.adapter.load_state_dict(adapter_state)
        self.model.head.load_state_dict(head_state)

    def get_adapter_head(self):
        return (
            copy.deepcopy(self.model.adapter.state_dict()),
            copy.deepcopy(self.model.head.state_dict())
        )

    def clone_model(self, model, target):
        for param, target_param in zip(model.parameters(), target.parameters()):
            target_param.data = param.data.clone()

    def update_parameters(self, model, new_params):
        for param, new_param in zip(model.parameters(), new_params):
            param.data = new_param.data.clone()

    def test_metrics(self):
        testloaderfull = self.load_test_data()
        self.model.eval()
        test_acc = 0
        test_num = 0

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for x, y in testloaderfull:
                if isinstance(x, list):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device).long()
                output = self.model(x)

                preds = torch.argmax(output, dim=1)
                test_acc += (preds == y).sum().item()
                test_num += y.shape[0]

                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(y.cpu().tolist())

        cm = confusion_matrix(all_labels, all_preds).tolist()  # convert to list for JSON compatibility
        print(f'{self.dataset_name}: {cm}')
        return test_acc, test_num, 0

    
    def train_metrics(self):
        trainloader = self.load_train_data()
        self.model.eval()

        train_num = 0
        losses = 0

        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device).long()
                output = self.model(x)
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        return losses, train_num

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



