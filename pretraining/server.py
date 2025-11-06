import torch
import os
import numpy as np
import h5py
import copy
import time
import random
from collections import defaultdict
import pickle
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from client import Client
from models.mae import MaskedAutoencoderViT
from tqdm import tqdm
import json
import torch
import yaml
from datetime import datetime
import pytz
from torch.optim import AdamW

import concurrent.futures

class Server(object):
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
        self.img_size = cfg['COMMON']['IMAGE_SIZE']
        self.patch_size = cfg['COMMON']['PATCH_SIZE']
        self.learning_rate = float(self.cfg['COMMON']['BASE_LR'])
        self.clients = []
        self.Budget = []
        self.global_model = eval(cfg['COMMON']['MODEL'])(img_size=self.img_size,patch_size=self.patch_size, hybrid=cfg['COMMON']['HYBRID']).to(self.device)
        self.eval_gap = cfg['SERVER']['EVAL_GAP']
        self.client_drop_rate = cfg['SERVER']['CLIENT_DROP_RATE']
        self.time_threthold = cfg['SERVER']['TIME_THRESHOLD']
        self.use_fedbn = cfg['SERVER']['USE_FEDBN']
        self.dataset_id = {0: 'NMC', 1: 'APTOS', 2: 'ODIR', 3: 'IDRiD', 4: 'MESSIDOR'}
        
        self.kst = pytz.timezone("Asia/Seoul")
        self.start_dt = datetime.now(self.kst)
        self.start_str = self.start_dt.strftime("%m%d_%H%M")
        self.folder = f"{self.start_str}/pretraining/"
        
        self.model_dir = os.path.join("results", self.folder)
        os.makedirs(self.model_dir, exist_ok=True)
        
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
        self.set_clients(Client)
        self.send_models()
        
        self.client_names = [self.dataset_id[i] for i in range(self.num_clients)]
        self.loss_history = {name: [] for name in self.client_names}
        self.loss_history['Average Loss'] = []
        self.loss_history['Average Std'] = []

        for i in range(self.global_rounds):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            
            round_losses = {}
            for client in self.selected_clients:
                # client.set_lr(current_lr)
                avg_loss = client.train(global_epoch=i)
                round_losses[client.dataset] = avg_loss
                self.loss_history[client.dataset].append(avg_loss)

            ordered_losses = [
                f"{round_losses.get(name, ' ---- '):.4f}" if name in round_losses else " ---- "
                for name in self.client_names
            ]
            
            rd = f"Round: {i} |"
            emt = " "
            print(f"Round: {i} |"," | ".join(self.client_names))
            print(f"{emt*(len(rd)-2)} |"," | ".join(ordered_losses))
            
            self.receive_models()
            self.aggregate_parameters()
            self.send_models()
            
            if i % self.eval_gap == 0:
                print(f"\n-------------- Round {i} Evaluation --------------")
                self.evaluate()

            if i % (self.eval_gap*10) == 0:
                print(f"\n-------------- Round {i} Saving Model --------------")
                self.save_model(i)
            
            self.Budget.append(time.time() - s_t)
            dash = '-' * 14
            print(f'{dash} time cost: {self.Budget[-1]:.2f}s {dash}')
            
        print(f"\n-------------- Final Evaluation --------------")
        self.evaluate()
        self.save_results()

    def evaluate(self):
        client_losses = []

        for client in self.clients:
            avg_loss = client.train_metrics()
            client_losses.append(avg_loss)

        total_avg_loss = np.mean(client_losses)
        std_loss = np.std(client_losses)

        print(f'Clients Average Loss: {total_avg_loss:.4f} ± {std_loss:.4f}')
        self.loss_history['Average Loss'].append(total_avg_loss)
        self.loss_history['Average Std'].append(std_loss)

    def send_models(self):
        assert len(self.clients) > 0

        bn_params = extract_bn_param_names(self.global_model)

        for client in self.clients:
            start_time = time.time()
            client_model = client.model

            if self.use_fedbn:
                client_param_dict = dict(client_model.named_parameters())

                for name, param in self.global_model.named_parameters():
                    if bn_params.get(name, False):
                        continue
                    client_param = client_param_dict[name]
                    client_param.data.copy_(param.data)
            else:
                client.set_parameters(self.global_model)

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
                
            tot_samples += client.train_samples
            self.uploaded_ids.append(client.id)
            self.uploaded_weights.append(client.train_samples)
            self.uploaded_models.append(client.model)
                
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)

        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data.zero_()
        
        if self.use_fedbn:
            for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
                self.add_parameters_except_bn(w, client_model)
        else:
            for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
                self.add_parameters(w, client_model)

    def add_parameters(self, w, client_model):
        for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w

            
    def add_parameters_except_bn(self, w, client_model):
        bn_params = extract_bn_param_names(self.global_model)

        client_param_dict = dict(client_model.named_parameters())

        for name, server_param in self.global_model.named_parameters():
            if bn_params.get(name, False):
                continue
            client_param = client_param_dict[name]
            server_param.data += client_param.data.clone() * w

    def save_model(self, rounds):
        model_dir = os.path.join("results", self.folder)
        os.makedirs(model_dir, exist_ok=True)
        
        round_dir = os.path.join(model_dir, f"global_params/{rounds}")
        os.makedirs(round_dir, exist_ok=True)
        
        model_filename = f"global_model.pt"
        model_path = os.path.join(round_dir, model_filename)
        torch.save(self.global_model.state_dict(), model_path)

        if self.use_fedbn:
            bn_dir = os.path.join(model_dir, f"bn_params/{rounds}")
            os.makedirs(bn_dir, exist_ok=True)
            
            bn_params = extract_bn_param_names(self.global_model)
            
            for client_id, client in enumerate(self.clients):
                bn_state_dict = {}
                for name, param in client.model.named_parameters():
                    if bn_params.get(name, False):
                        bn_state_dict[name] = param.data.clone()
                
                bn_filename = f"client_{client.dataset}_bn.pt"
                bn_path = os.path.join(bn_dir, bn_filename)
                torch.save(bn_state_dict, bn_path)
            
            print(f"✅ Global model and client-specific BN parameters saved in: {model_dir}")
        else:
            print(f"✅ Global model saved in: {model_dir}")


    def save_results(self):
        model_dir = os.path.join("results", self.folder)
        os.makedirs(model_dir, exist_ok=True)

        model_filename = f"global_model.pt"
        model_path = os.path.join(model_dir, model_filename)
        torch.save(self.global_model.state_dict(), model_path)
        
        if self.use_fedbn:
            bn_params = extract_bn_param_names(self.global_model)
            
            for client_id, client in enumerate(self.clients):
                bn_state_dict = {}
                for name, param in client.model.named_parameters():
                    if bn_params.get(name, False):
                        bn_state_dict[name] = param.data.clone()
                
                bn_filename = f"client_{client.dataset}_bn.pt"
                bn_path = os.path.join(model_dir, bn_filename)
                torch.save(bn_state_dict, bn_path)
            
            print(f"✅ Global model and client-specific BN parameters saved in: {model_dir}")
        else:
            print(f"✅ Global model saved in: {model_dir}")

        loss_filename = f"{self.algorithm}_{self.global_rounds}_loss_history.json"
        loss_path = os.path.join(model_dir, loss_filename)
        with open(loss_path, "w") as f:
            json.dump(self.loss_history, f, indent=4)

        cfg_filename = f"{self.algorithm}_{self.global_rounds}_config.yaml"
        cfg_path = os.path.join(model_dir, cfg_filename)
        with open(cfg_path, "w") as f:
            yaml.dump(self.cfg, f, default_flow_style=False)
            
        print(f"\n✅ Results saved in: {model_dir}")
        
        
def extract_bn_param_names(model):
    bn_params = {}
    for layer_name, layer in model.named_modules():
        if isinstance(layer, nn.BatchNorm2d):
            for param_name, _ in layer.named_parameters():
                full_name = f"{layer_name}.{param_name}"
                bn_params[full_name] = True
    return bn_params

