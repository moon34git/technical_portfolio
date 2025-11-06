import random
import time
import numpy as np
from clientrep import clientRep
from threading import Thread
from collections import defaultdict
import os
import json
import torch
import yaml
import pytz
from datetime import datetime
from tqdm import tqdm
from models.mae import MaskedAutoencoderViT
from models.classifier import *
from data.dataset_utils import *
import copy

class FedRep(object):
    def __init__(self, cfg, times):

        self.cfg = cfg
        self.device = cfg['COMMON']['DEVICE']
        self.num_classes = cfg['COMMON']['NUM_CLASSES']
        self.global_rounds = cfg['SERVER']['GLOBAL_ROUNDS']
        self.num_clients = cfg['SERVER']['NUM_CLIENTS']
        self.join_ratio = cfg['SERVER']['JOIN_RATIO']
        self.num_join_clients = int(self.num_clients * self.join_ratio)
        self.algorithm = cfg['SERVER']['ALGORITHM']
        self.goal = cfg['SERVER']['GOAL']
        self.img_size = cfg['COMMON']['IMG_SIZE']
        self.patch_size = cfg['COMMON']['PATCH_SIZE']
        self.model_path = cfg['COMMON']['MODEL_PATH']
        self.adapft = cfg['COMMON']['ADAPFT']
        self.clients = []
        self.Budget = []
        print(self.cfg)

        self.init_model()

        self.eval_gap = cfg['SERVER']['EVAL_GAP']
        self.client_drop_rate = cfg['SERVER']['CLIENT_DROP_RATE']
        self.time_threthold = cfg['SERVER']['TIME_THRESHOLD']

        # self.dataset_id = {0: 'NMC', 1: 'APTOS', 2: 'ODIR', 3: 'IDRiD', 4: 'MESSIDOR'}
        self.dataset_id = {i: name for i, name in enumerate(cfg['CLIENTS'].keys())}
        print(self.dataset_id)

        self.acc_list = []
        self.train_loss_list = []
        
        self.kst = pytz.timezone("Asia/Seoul")
        self.start_dt = datetime.now(self.kst)
        self.start_str = self.start_dt.strftime("%m%d_%H%M")
        self.folder = f"{self.start_str}/"
        
        self.model_dir = os.path.join("results", self.folder)
        os.makedirs(self.model_dir, exist_ok=True)

        self.set_clients(clientRep)
        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

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
            self.global_model = FineTunedMAE_Shallow(pretrained_mae, num_classes=self.num_classes, freeze=True).to(self.device)
            self.update_params = list(self.global_model.adapter.parameters()) + \
              list(self.global_model.head.parameters())
            print("Adapter Fine-tuning")
        else:
            self.global_model = FineTunedMAE(pretrained_mae, num_classes=self.num_classes, freeze=False).to(self.device)
            self.update_params = list(self.global_model.parameters())
            print("No Adapter, Full Fine-tuning")

    def set_clients(self, clientObj):
        for i in range(self.num_clients):
            dataset_name = self.dataset_id[i]
            img_path=self.cfg['CLIENTS'][dataset_name]['IMAGE_PATH']
            csv_path=self.cfg['CLIENTS'][dataset_name]['CSV_PATH']            
            client = clientObj(cfg=self.cfg, 
                               id=i, 
                               dataset=dataset_name,
                               img_path=img_path,
                               csv_path=csv_path)
            self.clients.append(client)


    def select_clients(self):
        self.current_num_join_clients = self.num_join_clients
        selected_clients = list(np.random.choice(self.clients, self.current_num_join_clients, replace=False))
        return selected_clients

    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()

            if self.adapft:
                self.send_adapter()
            else:
                self.send_models()

            if i != 0 and i % self.eval_gap == 0:
                print(f"\n------------- Round {i} -------------")
                print("\nEvaluate personalized models")
                self.evaluate_and_save_model(rnd=i)
                    
            for client in tqdm(self.selected_clients):
                client.train()

            if self.adapft:
                self.receive_adapter()
                self.aggregate_adapter()
            else:
                self.receive_models()
                self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
        
        print('-----------------Save Evaluation Results-----------------')
        save_evaluation_results(
            cfg=self.cfg,
            save_dir=self.model_dir,
            test_accs=self.acc_list,           
            train_losses=self.train_loss_list
        )


    def evaluate_and_save_model(self, rnd, acc=None, loss=None):
        test_accs, test_nums, test_corrects = [], [], []
        train_losses, train_nums = [], []

        print(f"\n📊 [Evaluation at Round {len(self.acc_list)}]")

        for client in tqdm(self.clients):  
            acc, total, _ = client.test_metrics()
            loss, num = client.train_metrics()

            test_accs.append(acc / total)
            test_nums.append(total)
            test_corrects.append(acc)

            train_losses.append(loss)
            train_nums.append(num)

            # if rnd % 50 == 0:
            #     client.save_model(rnd=rnd, save_dir=os.path.join(self.model_dir, "models"))
            client.save_model(rnd=rnd, save_dir=os.path.join(self.model_dir, "models"))


        avg_test_acc = sum(test_corrects) / sum(test_nums)
        avg_train_loss = sum(train_losses) / sum(train_nums)
        std_test_acc = np.std(test_accs)

        self.acc_list.append(avg_test_acc)
        self.train_loss_list.append(avg_train_loss)

        print(f"🔵 Avg Train Loss: {avg_train_loss:.4f}")
        print(f"🟢 Avg Test Acc: {avg_test_acc:.4f} | Std: {std_test_acc:.4f}")
        print("🔴 " + " | ".join([f"Acc{i+1}: {acc:.4f}" for i, acc in enumerate(test_accs)]))


    def send_models(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()
            client.set_parameters(self.global_model)
            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def send_adapter(self):
        assert len(self.clients) > 0

        for client in self.clients:
            start_time = time.time()
            adapter_state = self.global_model.adapter.state_dict()
            client.set_adapter(adapter_state)
            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)


    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1-self.client_drop_rate) * self.current_num_join_clients))

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0
        for client in active_clients:
            try:
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                        client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples
                self.uploaded_ids.append(client.id)
                self.uploaded_weights.append(client.train_samples)
                self.uploaded_models.append(client.model)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def receive_adapter(self):
        assert len(self.selected_clients) > 0

        active_clients = random.sample(
            self.selected_clients, int((1 - self.client_drop_rate) * self.current_num_join_clients))

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_adapters = []

        tot_samples = 0
        for client in active_clients:
            try:
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                                client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            except ZeroDivisionError:
                client_time_cost = 0

            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples
                self.uploaded_ids.append(client.id)
                self.uploaded_weights.append(client.train_samples)
                adapter_state = client.get_adapter()
                self.uploaded_adapters.append(adapter_state)

        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples


    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)

        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data.zero_()
            
        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w, client_model)

    def aggregate_adapter(self):
        assert len(self.uploaded_adapters) > 0

        # Adapter aggregation
        new_adapter = copy.deepcopy(self.global_model.adapter.state_dict())
        for key in new_adapter:
            new_adapter[key] = sum(
                w * client_adapter[key] for w, client_adapter in zip(self.uploaded_weights, self.uploaded_adapters)
            )

        # Apply
        self.global_model.adapter.load_state_dict(new_adapter)

    def add_parameters(self, w, client_model):
        for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w

def save_evaluation_results(cfg, save_dir, test_accs, train_losses):
    os.makedirs(save_dir, exist_ok=True)
    eval_dict = {
        "rounds": list(range(len(test_accs))),
        "test_accuracy": [float(f"{x:.4f}") for x in test_accs],
        "train_loss": [float(f"{x:.4f}") for x in train_losses],
    }

    save_path = os.path.join(save_dir, "evaluation_results.json")
    with open(save_path, "w") as f:
        json.dump(eval_dict, f, indent=4)
        
    print(f"📁 Evaluation results saved to {save_path}")


