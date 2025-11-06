import torch.nn.functional as F
import time
import numpy as np
from clientproto import clientProto
from threading import Thread
from collections import defaultdict
import os
import json
import torch
import yaml
import pytz
from datetime import datetime
from tqdm import tqdm

class FedProto(object):
    def __init__(self, cfg, times):

        self.cfg = cfg
        self.device = cfg['COMMON']['DEVICE']
        self.num_classes = cfg['COMMON']['NUM_CLASSES']
        self.global_protos = [None for _ in range(self.num_classes)]
        self.global_rounds = cfg['SERVER']['GLOBAL_ROUNDS']
        self.num_clients = cfg['SERVER']['NUM_CLIENTS']
        self.join_ratio = cfg['SERVER']['JOIN_RATIO']
        self.num_join_clients = int(self.num_clients * self.join_ratio)
        self.algorithm = cfg['SERVER']['ALGORITHM']
        self.goal = cfg['SERVER']['GOAL']
        self.img_size = cfg['COMMON']['IMG_SIZE']
        self.patch_size = cfg['COMMON']['PATCH_SIZE']
        self.qa = cfg['COMMON']['QA']
        self.tau = cfg['COMMON']['TAU']
        self.alpha = cfg['COMMON']['ALPHA']
        self.clients = []
        self.Budget = []

        self.eval_gap = cfg['SERVER']['EVAL_GAP']
        self.client_drop_rate = cfg['SERVER']['CLIENT_DROP_RATE']
        self.time_threthold = cfg['SERVER']['TIME_THRESHOLD']

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
        print(self.cfg)

        self.set_clients(clientProto)
        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

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

            if i != 0 and i % self.eval_gap == 0:
                print(f"\n------------- Round {i} -------------")
                print("\nEvaluate personalized models")
                self.evaluate_and_save_model(rnd=i)

                print("\nSave Global Prototypes")
                save_global_protos_npz(
                    self.global_protos,
                    save_dir=os.path.join(self.model_dir, "global_prototypes"),
                    round_num=i
                )
                    
            for client in tqdm(self.selected_clients):
                client.train()

            self.receive_protos()
            if self.qa:
                self.global_protos = variance_weighted_proto_aggregation(
                    self.uploaded_proto_stats, i,
                    tau=self.tau,
                    alpha=self.alpha,
                    epsilon=1e-8
                )
            else:
                self.global_protos = proto_aggregation(self.uploaded_protos)
            self.send_protos()

            self.Budget.append(time.time() - s_t)
        
        print('-----------------Save Evaluation Results-----------------')
        save_evaluation_results(
            cfg=self.cfg,
            save_dir=self.model_dir,
            test_accs=self.acc_list,           
            train_losses=self.train_loss_list
        )

    def send_protos(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()

            client.set_protos(self.global_protos)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_protos(self):
        assert (len(self.selected_clients) > 0)

        self.uploaded_ids = []
        self.uploaded_protos = []
        self.uploaded_proto_stats = []
        for client in self.selected_clients:
            self.uploaded_ids.append(client.id)
            self.uploaded_protos.append(client.protos)
            proto_stats = client.collect_protos_with_stats()
            self.uploaded_proto_stats.append(proto_stats)

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


            client.save_model(rnd=rnd, save_dir=os.path.join(self.model_dir, "models"))

        avg_test_acc = sum(test_corrects) / sum(test_nums)
        avg_train_loss = sum(train_losses) / sum(train_nums)
        std_test_acc = np.std(test_accs)

        self.acc_list.append(avg_test_acc)
        self.train_loss_list.append(avg_train_loss)

        print(f"🔵 Avg Train Loss: {avg_train_loss:.4f}")
        print(f"🟢 Avg Test Acc: {avg_test_acc:.4f} | Std: {std_test_acc:.4f}")
        print("🔴 " + " | ".join([f"Acc{i+1}: {acc:.4f}" for i, acc in enumerate(test_accs)]))


def variance_weighted_proto_aggregation(local_proto_stats_list, round_num, tau, alpha, epsilon=1e-6):
    if round_num % 10 == 0:
        print(f"\n🔄 [VARIANCE-WEIGHTED AGGREGATION] Starting weighted aggregation...")
        print(f"   📊 Clients: {len(local_proto_stats_list)}, ε: {epsilon}")
    
    agg_protos = defaultdict(list)
    agg_weights = defaultdict(list)

    for client_idx, client_stats in enumerate(local_proto_stats_list):
        for cls, stats in client_stats.items():
            proto = stats["proto"]
            var   = stats["var"]
            count = stats["count"]

            raw_weight = count / (var + epsilon)

            agg_protos[cls].append(proto)
            agg_weights[cls].append(raw_weight)
        

    global_protos = {}
    for cls in agg_protos.keys():
        weights = torch.tensor(agg_weights[cls], device=agg_protos[cls][0].device)
        if round_num % 10 == 0:
            print(f"   📈 Class {cls} - Raw weights: {weights.tolist()}")
        
        max_weight_ratio = tau
        weights_normalized = weights / weights.sum() 
        
        max_idx = torch.argmax(weights_normalized)
        max_weight = weights_normalized[max_idx]
        
        if max_weight > max_weight_ratio:
            weights_normalized[max_idx] = max_weight_ratio
            
            remaining_weight = 1.0 - max_weight_ratio
            other_weights_sum = weights_normalized.sum() - max_weight_ratio
            
            for i in range(len(weights_normalized)):
                if i != max_idx:
                    weights_normalized[i] = weights_normalized[i] / other_weights_sum * remaining_weight

        if round_num % 10 == 0:
            print(f"   📊 Class {cls} - Capped & Redistributed weights: {weights_normalized.tolist()}")
        weights = weights_normalized

        weighted_proto = torch.sum(
            torch.stack([w * p for w, p in zip(weights, agg_protos[cls])]), dim=0
        )

        simple_proto = torch.mean(torch.stack(agg_protos[cls]), dim=0)

        ema_alpha = alpha
        print(f"   🎯 EMA: alpha={ema_alpha}")
        final_proto = alpha * weighted_proto + (1 - alpha) * simple_proto
        
        global_protos[cls] = final_proto.detach()

        avg_weight = weights.mean().item()
        
        if round_num % 10 == 0:
            print(f"   🎯 Class {cls}: Avg weight={avg_weight:.4f}, Clients={len(weights)}")


    if round_num % 10 == 0:
        print(f"   ✅ Variance-weighted aggregation completed!\n")
    return global_protos


def save_global_protos_npz(global_protos, save_dir, round_num=None):
    os.makedirs(save_dir, exist_ok=True)

    npz_dict = {
        f'class_{int(k)}': v.cpu().numpy() if hasattr(v, 'cpu') else v
        for k, v in global_protos.items()
    }

    file_name = f"global_protos_round{round_num}.npz" if round_num is not None else "global_protos.npz"
    save_path = os.path.join(save_dir, file_name)
    np.savez_compressed(save_path, **npz_dict)

    print(f"✅ Prototypes saved in NPZ format: {save_path}")


def save_evaluation_results(cfg, save_dir, test_accs, train_losses):
    os.makedirs(save_dir, exist_ok=True)
    eval_dict = {
        "rounds": list(range(len(test_accs))),
        "test_accuracy": [float(f"{x:.4f}") for x in test_accs],
        "train_loss": [float(f"{x:.4f}") for x in train_losses],
        "LAMBDA": cfg['COMMON']['LDA'],
        "BETA": cfg['COMMON']['BETA'],
        "Loss Mode": cfg['COMMON']['LOSS_TYPE']
    }

    save_path = os.path.join(save_dir, "evaluation_results.json")
    with open(save_path, "w") as f:
        json.dump(eval_dict, f, indent=4)
        
    print(f"📁 Evaluation results saved to {save_path}")


def proto_aggregation(local_protos_list):
    agg_protos = defaultdict(list)
    for local_protos in local_protos_list:
        for label in local_protos.keys():
            agg_protos[label].append(local_protos[label])

    for [label, proto_list] in agg_protos.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            agg_protos[label] = proto / len(proto_list)
        else:
            agg_protos[label] = proto_list[0].data

    return agg_protos
