import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm
from collections import defaultdict, Counter
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import numpy as np

from data.dataset_utils import *
from models.mae import MaskedAutoencoderViT
from models.classifier import FineTunedMAE_Shallow

import argparse
import json
import yaml
import time
from pathlib import Path
from utils import fix_seeds, setup_cudnn


def load_config_and_setup(config_path, device_id='cuda:0'):
    fix_seeds(34)
    
    device = torch.device(device_id)
    
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    
    return cfg, device


def load_data(cfg, device):
    transform = transforms.Compose([
        transforms.Resize(size=cfg['IMG_SIZE']),
        transforms.CenterCrop(size=(cfg['IMG_SIZE'], cfg['IMG_SIZE'])), 
        transforms.ToTensor(),
        transforms.Normalize(cfg['MEAN'], cfg['STD'])
    ])
    
    train_dataset = eval(cfg['DATASET'])(
        cfg['IMG_DIR'], cfg['CSV_DIR'], 
        train_ratio=cfg['TRAIN_RATIO'], 
        train=True, 
        transform=transform
    )
    train_loader = DataLoader(train_dataset, batch_size=cfg['BATCH_SIZE'], shuffle=False, num_workers=4)
    
    test_dataset = eval(cfg['DATASET'])(
        cfg['IMG_DIR'], cfg['CSV_DIR'], 
        train_ratio=cfg['TRAIN_RATIO'], 
        train=False, 
        transform=transform
    )
    test_loader = DataLoader(test_dataset, batch_size=cfg['BATCH_SIZE'], shuffle=False, num_workers=4)
    
    combined_dataset = ConcatDataset([train_dataset, test_dataset])
    combined_loader = DataLoader(combined_dataset, batch_size=cfg['BATCH_SIZE'], shuffle=False, num_workers=4)
    
    return train_loader, test_loader, combined_loader


def load_models(cfg, device):
    encoder_pt = torch.load(cfg['encoder_path'], map_location=device)
    
    pretrained_mae = MaskedAutoencoderViT(
        img_size=cfg['IMG_SIZE'],
        patch_size=cfg['PATCH_SIZE'],
        hybrid=False
    ).to(device)
    
    if isinstance(encoder_pt, nn.Module):
        encoder_pt = encoder_pt.state_dict()
    
    pretrained_mae.load_state_dict(encoder_pt, strict=True)
    pretrained_mae.eval()
    
    return pretrained_mae


def build_client_model(client_model, encoder, cfg, device):
    model = FineTunedMAE_Shallow(encoder, num_classes=cfg['NUM_CLASSES'], freeze=True).to(device)
    
    if isinstance(client_model, nn.Module):
        client_model = client_model.state_dict()
        
        adapter_state_dict = {
            k.replace("adapter.", ""): v for k, v in client_model.items() if k.startswith("adapter.")
        }
        
        head_state_dict = {
            k.replace("head.", ""): v for k, v in client_model.items() if k.startswith("head.")
        }
        
        model.adapter.load_state_dict(adapter_state_dict)
        model.head.load_state_dict(head_state_dict)
    else:
        model.adapter.load_state_dict(client_model['adapter'])
        model.head.load_state_dict(client_model['head'])
    
    model.eval()
    return model


def extract_all_clients_feature_entropy(client_models, dataloader, device):
    client_feature_dicts = {name: defaultdict(list) for name in client_models}

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(tqdm(dataloader)):
            x, y = x.to(device), y.long().to(device)
            batch_size = x.size(0)

            for name, model in client_models.items():
                feats = model.extract_representation(x)
                logits = model.head(feats)
                probs = F.softmax(logits, dim=1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)

                for i in range(batch_size):
                    class_id = y[i].item()
                    pred_label = logits[i].argmax().item()
                    
                    client_feature_dicts[name][class_id].append({
                        "feature": feats[i].cpu(),
                        "entropy": entropy[i].item(),
                        "image": x[i].cpu(),
                        "label": class_id,
                        "pred_label": pred_label,
                        "logits": logits[i].cpu()
                    })

    return client_feature_dicts


def calculate_entropy_thresholds(client_feature_dicts, percentile):
    thresholds = {}
    
    for client_name, feature_dict in client_feature_dicts.items():
        all_entropies = [
            entry["entropy"]
            for entries in feature_dict.values()
            for entry in entries
        ]
        ent = np.array(all_entropies)
        
        thresholds[client_name] = np.percentile(ent, percentile)
    
    return thresholds


def intersect_filtered_samples(client_feature_dicts, client_thresholds, policy='strict', min_count=3):
    common_filtered_dict = defaultdict(list)
    client_names = list(client_feature_dicts.keys())
    class_ids = client_feature_dicts[client_names[0]].keys()

    for class_id in class_ids:
        client_entries_list = [client_feature_dicts[client][class_id] for client in client_names]
        n_samples = len(client_entries_list[0])

        for i in range(n_samples):
            pass_count = 0
            for client_idx, client in enumerate(client_names):
                if client_entries_list[client_idx][i]['entropy'] < client_thresholds[client]:
                    pass_count += 1

            if (policy == 'strict' and pass_count == len(client_names)) or \
               (policy == 'majority' and pass_count >= len(client_names)//2 + 1) or \
               (policy == 'relaxed' and pass_count >= min_count):
                common_filtered_dict[class_id].append(client_entries_list[0][i])

    return common_filtered_dict


def evaluate_voting(client_feature_dicts, common_dict):
    client_names = list(client_feature_dicts.keys())
    y_true, y_pred = [], []

    for class_id, sample_list in common_dict.items():
        for i in range(len(sample_list)):
            gt_label = sample_list[i]['label']
            y_true.append(gt_label)

            votes = [
                client_feature_dicts[client][class_id][i]['pred_label']
                for client in client_names
            ]

            vote_result = Counter(votes).most_common(1)[0][0]
            y_pred.append(vote_result)

    print("\n🎯 Overall Accuracy:")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")

    print("\n📊 Classification Report:")
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))

    print("\n🧩 Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    
    if y_true:
        macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'macro_f1': macro_f1,
            'weighted_f1': weighted_f1,
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
            'num_samples': len(y_true)
        }
    return None


def main():
    parser = argparse.ArgumentParser(description='Classifier-based Collaborative Inference')
    parser.add_argument('--config', type=str)
    parser.add_argument('--device', type=str)
    args = parser.parse_args()
    
    print("=" * 80)
    print("🚀 Starting Classifier-based Collaborative Inference")
    print("=" * 80)
    cfg, device = load_config_and_setup(args.config, args.device)
    
    print("\n📦 Loading data...")
    train_loader, test_loader, combined_loader = load_data(cfg, device)
    
    print("\n🔧 Loading models...")
    pretrained_mae = load_models(cfg, device)

    
    if cfg['fed_algo'] == 'fedavg':
        print("\n🏗️ Building client models... (FedAvg Scenario)")
        combined_pt = torch.load(cfg['combined_model'], map_location=device)

        client_models = {
            "fedavg": build_client_model(combined_pt, pretrained_mae, cfg, device)
        }

        kappa = 1

    else:
        print("\n🏗️ Building client models...")
        aptos_pt = torch.load(cfg['aptos_model'], map_location=device)
        odir_pt = torch.load(cfg['odir_model'], map_location=device)
        idrid_pt = torch.load(cfg['idrid_model'], map_location=device)
        mes_pt = torch.load(cfg['messidor_model'], map_location=device)
        
        client_models = {
            "aptos": build_client_model(aptos_pt, pretrained_mae, cfg, device),
            "odir": build_client_model(odir_pt, pretrained_mae, cfg, device),
            "idrid": build_client_model(idrid_pt, pretrained_mae, cfg, device),
            "messidor": build_client_model(mes_pt, pretrained_mae, cfg, device)
        }

        kappa = cfg['KAPPA']
    
    if cfg['scenario'] == 'unlabeled':
        print("\n🎯 Extracting features and predictions... (Unlabeled Scenario)")
        client_feature_dicts = extract_all_clients_feature_entropy(client_models, test_loader, device)
    else:
        print("\n🎯 Extracting features and predictions... (Late-joining Scenario)")
        client_feature_dicts = extract_all_clients_feature_entropy(client_models, combined_loader, device)
    
    print("\n📊 Calculating entropy thresholds...")
    thresholds = calculate_entropy_thresholds(client_feature_dicts, cfg['ETA'])
    
    print(f"\nEntropy Thresholds (Percentile: {cfg['ETA']}%):")
    for client_name, threshold in thresholds.items():
        print(f"  {client_name.upper()}: {threshold:.4f}")
    
    print(f"\n🔍 Filtering with policy '{cfg['policy']}' and min_count={cfg['KAPPA']}...")
    common_dict = intersect_filtered_samples(
        client_feature_dicts, 
        thresholds, 
        policy=cfg['policy'], 
        min_count=kappa
    )
    
    total_samples = sum(len(samples) for samples in common_dict.values())
    print(f"\n📈 Filtered samples: {total_samples}")
    print(f"   Class 0: {len(common_dict[0])}")
    print(f"   Class 1: {len(common_dict[1])}")
    
    print("\n🗳️ Evaluating collaborative voting...")
    results = evaluate_voting(client_feature_dicts, common_dict)
    
    print("\n" + "=" * 80)
    print("✅ Classifier-based Collaborative Inference Completed!")
    print("=" * 80)
    
    return results, client_feature_dicts, common_dict


if __name__ == "__main__":
    results, client_feature_dicts, common_dict = main()

