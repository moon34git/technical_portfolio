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
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, accuracy_score
import numpy as np

from data.dataset_utils import *
from models.mae import MaskedAutoencoderViT
from models.classifier import FineTunedMAE_Shallow

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

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
    
    gp = np.load(cfg['global_protos'])
    
    return pretrained_mae, gp


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


def extract_all_clients_feature_entropy_with_gp(client_models, dataloader, device, gp):
    client_feature_dicts = {name: defaultdict(list) for name in client_models}
    
    gp_tensors = {}
    for gp_class, gp_proto in gp.items():
        if isinstance(gp_proto, np.ndarray):
            gp_tensors[gp_class] = torch.tensor(gp_proto, dtype=torch.float32).to(device)
        else:
            gp_tensors[gp_class] = gp_proto.to(device)
    
    print(f"🎯 GP loaded for distance-based prediction: {list(gp_tensors.keys())}")

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(tqdm(dataloader)):
            x, y = x.to(device), y.long().to(device)
            batch_size = x.size(0)

            for name, model in client_models.items():
                model.eval()
                feats = model.extract_representation(x)

                for i in range(batch_size):
                    class_id = y[i].item()
                    
                    feature = feats[i]
                    
                    dist_to_class_0 = torch.norm(feature - gp_tensors['class_0'], p=2).item()
                    dist_to_class_1 = torch.norm(feature - gp_tensors['class_1'], p=2).item()
                    
                    gp_pred_label = 0 if dist_to_class_0 < dist_to_class_1 else 1
                    
                    client_feature_dicts[name][class_id].append({
                        "feature": feats[i].cpu(),
                        "image": x[i].cpu(),
                        "label": class_id,
                        "pred_label": gp_pred_label,
                        "distance_to_class_0": dist_to_class_0,
                        "distance_to_class_1": dist_to_class_1
                    })

    return client_feature_dicts


def add_gp_distances_to_client_dicts_simple(client_feature_dicts, gp, device):
    gp_tensors = {}
    for gp_class, gp_proto in gp.items():
        if isinstance(gp_proto, np.ndarray):
            gp_tensors[gp_class] = torch.tensor(gp_proto, dtype=torch.float32).to(device)
        else:
            gp_tensors[gp_class] = gp_proto.to(device)
    
    print(f"🎯 GP Classes: {list(gp_tensors.keys())}")
    
    for client_name, feature_dict in client_feature_dicts.items():
        print(f"🔍 Adding GP distances for {client_name.upper()}")
        
        for class_id, samples in feature_dict.items():
            for sample in tqdm(samples, desc=f"{client_name} class {class_id}"):
                feature = sample['feature'].to(device)
                
                sample['distance_to_class_0'] = torch.norm(feature - gp_tensors['class_0'], p=2).item()
                sample['distance_to_class_1'] = torch.norm(feature - gp_tensors['class_1'], p=2).item()
        
        print(f"  ✅ {client_name.upper()} completed")
    
    return client_feature_dicts

def apply_zscore_normalization(client_feature_dicts):
    print("🔧 Applying Z-score Normalization")
    print("=" * 60)
    
    normalized_client_dicts = {}
    normalization_stats = {}
    
    for client_name, feature_dict in client_feature_dicts.items():
        print(f"\n📊 Processing {client_name.upper()}")
        
        all_dist_class0 = []
        all_dist_class1 = []
        
        for samples in feature_dict.values():
            for sample in samples:
                all_dist_class0.append(sample['distance_to_class_0'])
                all_dist_class1.append(sample['distance_to_class_1'])
        
        mean_class0 = np.mean(all_dist_class0)
        std_class0 = np.std(all_dist_class0)
        mean_class1 = np.mean(all_dist_class1)
        std_class1 = np.std(all_dist_class1)
        
        normalization_stats[client_name] = {
            'class0_mean': mean_class0,
            'class0_std': std_class0,
            'class1_mean': mean_class1,
            'class1_std': std_class1
        }
        
        print(f"  Class 0 - Mean: {mean_class0:.4f}, Std: {std_class0:.4f}")
        print(f"  Class 1 - Mean: {mean_class1:.4f}, Std: {std_class1:.4f}")
        
        normalized_dict = defaultdict(list)
        
        for class_id, samples in feature_dict.items():
            for sample in samples:
                normalized_sample = sample.copy()
                
                zscore_class0 = (sample['distance_to_class_0'] - mean_class0) / std_class0
                zscore_class1 = (sample['distance_to_class_1'] - mean_class1) / std_class1
                
                normalized_sample['zscore_to_class_0'] = zscore_class0
                normalized_sample['zscore_to_class_1'] = zscore_class1
                
                if zscore_class0 < zscore_class1:
                    normalized_sample['nearest_class_zscore'] = 0
                    normalized_sample['nearest_zscore'] = zscore_class0
                    normalized_sample['zscore_margin'] = zscore_class1 - zscore_class0
                else:
                    normalized_sample['nearest_class_zscore'] = 1
                    normalized_sample['nearest_zscore'] = zscore_class1
                    normalized_sample['zscore_margin'] = zscore_class0 - zscore_class1
                
                normalized_dict[class_id].append(normalized_sample)
        
        normalized_client_dicts[client_name] = normalized_dict
        
        all_zscore_0 = []
        all_zscore_1 = []
        for samples in normalized_dict.values():
            for sample in samples:
                all_zscore_0.append(sample['zscore_to_class_0'])
                all_zscore_1.append(sample['zscore_to_class_1'])
        
        print(f"  Normalized Class 0 - Mean: {np.mean(all_zscore_0):.4f}, Std: {np.std(all_zscore_0):.4f}")
        print(f"  Normalized Class 1 - Mean: {np.mean(all_zscore_1):.4f}, Std: {np.std(all_zscore_1):.4f}")
    
    print("\n" + "=" * 60)
    print("✅ Z-score normalization completed!")
    
    return normalized_client_dicts, normalization_stats


def filter_by_zscore_threshold(normalized_client_dicts, threshold=0):
    filtered_results = {}
    
    print(f"🔍 Filtering with Z-score threshold: {threshold}")
    print("=" * 60)
    
    for client_name, client_dict in normalized_client_dicts.items():
        print(f"\n📊 Filtering {client_name.upper()}")
        
        filtered_dict = defaultdict(list)
        
        class0_filtered = 0
        class1_filtered = 0
        
        for class_id, samples in client_dict.items():
            original_class_count = len(samples)
            
            for sample in samples:
                close_to_class0 = sample['zscore_to_class_0'] <= threshold
                close_to_class1 = sample['zscore_to_class_1'] <= threshold
                
                if close_to_class0 or close_to_class1:
                    sample['close_to_class0'] = close_to_class0
                    sample['close_to_class1'] = close_to_class1
                    
                    if close_to_class0 and close_to_class1:
                        sample['filter_reason'] = 'both_close'
                    elif close_to_class0:
                        sample['filter_reason'] = 'close_to_class0'
                    else:
                        sample['filter_reason'] = 'close_to_class1'
                    
                    filtered_dict[class_id].append(sample)
                    
                    if class_id == 0:
                        class0_filtered += 1
                    else:
                        class1_filtered += 1
            
            filtered_class_count = len(filtered_dict[class_id])
            print(f"  Class {class_id}: {filtered_class_count}/{original_class_count} samples ({100*filtered_class_count/original_class_count:.1f}%)")
        
        original_total = sum(len(samples) for samples in client_dict.values())
        filtered_total = sum(len(samples) for samples in filtered_dict.values())
        
        print(f"  📈 Total: {filtered_total}/{original_total} samples ({100*filtered_total/original_total:.1f}%)")
        print(f"  📋 Class 0 filtered: {class0_filtered}, Class 1 filtered: {class1_filtered}")
        
        filtered_results[client_name] = filtered_dict
    
    print("\n" + "=" * 60)
    print("✅ Z-score filtering completed!")
    
    return filtered_results


def determine_predicted_class(sample):
    if sample['zscore_to_class_0'] < sample['zscore_to_class_1']:
        return 0
    else:
        return 1


def find_consensus_intersection_samples(filtered_client_dicts, min_clients=2):
    print(f"🔍 Finding Consensus Intersection (min_clients: {min_clients})")
    print("=" * 70)
    
    sample_groups = defaultdict(list)
    
    for client_name, client_dict in filtered_client_dicts.items():
        global_idx = 0
        
        for class_id in sorted(client_dict.keys()):
            for sample in client_dict[class_id]:
                predicted_class = determine_predicted_class(sample)
                
                sample_groups[global_idx].append({
                    'client': client_name,
                    'sample': sample,
                    'predicted_class': predicted_class,
                    'true_label': sample['label'],
                    'original_class_id': class_id
                })
                
                global_idx += 1
    
    results = {}
    
    for required_clients in range(2, 5):
        consensus_samples = []
        
        for global_idx, client_predictions in sample_groups.items():
            if len(client_predictions) < required_clients:
                continue
            
            predicted_classes = [pred['predicted_class'] for pred in client_predictions]
            unique_predictions = set(predicted_classes)
            
            if len(unique_predictions) == 1:
                consensus_class = predicted_classes[0]
                true_label = client_predictions[0]['true_label']
                
                consensus_samples.append({
                    'global_idx': global_idx,
                    'num_clients': len(client_predictions),
                    'clients': [pred['client'] for pred in client_predictions],
                    'consensus_class': consensus_class,
                    'true_label': true_label,
                    'is_correct': consensus_class == true_label,
                    'client_predictions': client_predictions
                })
        
        results[required_clients] = consensus_samples
        
        correct_count = sum(1 for s in consensus_samples if s['is_correct'])
        total_count = len(consensus_samples)
        accuracy = 100 * correct_count / total_count if total_count > 0 else 0
        
        print(f"📊 {required_clients}+ clients consensus:")
        print(f"   Total samples: {total_count}")
        print(f"   Correct predictions: {correct_count}/{total_count} ({accuracy:.1f}%)")

        y_true = [s['true_label'] for s in consensus_samples]
        y_pred = [s['consensus_class'] for s in consensus_samples]

        if y_true:
            macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
            weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

            print(f"   Macro F1-score: {macro_f1:.4f}")
            print(f"   Weighted F1-score: {weighted_f1:.4f}")
        else:
            print("   No samples to evaluate F1-scores.")

        class_counts = defaultdict(int)
        class_correct = defaultdict(int)
        
        for sample in consensus_samples:
            pred_class = sample['consensus_class']
            class_counts[pred_class] += 1
            if sample['is_correct']:
                class_correct[pred_class] += 1
        
        print(f"   Class distribution:")
        for cls in sorted(class_counts.keys()):
            cls_acc = 100 * class_correct[cls] / class_counts[cls] if class_counts[cls] > 0 else 0
            print(f"     Class {cls}: {class_counts[cls]} samples ({cls_acc:.1f}% accuracy)")
        print()
    
    return results


def refine_with_zscore_margin_filtering(consensus_samples, filtered_client_dicts, top_percentage=25, min_margin_clients=3):
    print("🔍 Z-score Margin-based Refinement")
    print("=" * 60)
    
    zscore_margin_thresholds = {}
    
    for client_name, client_dict in filtered_client_dicts.items():
        all_zscore_margins = []
        for samples in client_dict.values():
            for sample in samples:
                if 'zscore_to_class_0' in sample and 'zscore_to_class_1' in sample:
                    zscore_margin = abs(sample['zscore_to_class_0'] - sample['zscore_to_class_1'])
                    all_zscore_margins.append(zscore_margin)
        
        if all_zscore_margins:
            all_zscore_margins.sort(reverse=True)
            threshold_idx = int(len(all_zscore_margins) * top_percentage / 100)
            threshold = all_zscore_margins[threshold_idx] if threshold_idx < len(all_zscore_margins) else all_zscore_margins[-1]
            zscore_margin_thresholds[client_name] = threshold
            
            print(f"📊 {client_name.upper()} zscore margin threshold (top {top_percentage}%): {threshold:.4f}")
    
    zscore_margin_refined_samples = []
    
    for consensus_sample in consensus_samples:
        client_predictions = consensus_sample['client_predictions']
        
        zscore_margin_pass_count = 0
        total_clients = len(client_predictions)
        zscore_margin_info = {}
        
        for pred_info in client_predictions:
            client = pred_info['client']
            sample = pred_info['sample']
            
            if 'zscore_to_class_0' in sample and 'zscore_to_class_1' in sample:
                zscore_margin = abs(sample['zscore_to_class_0'] - sample['zscore_to_class_1'])
                threshold = zscore_margin_thresholds.get(client, 0)
                
                zscore_margin_info[client] = {
                    'zscore_margin': zscore_margin,
                    'threshold': threshold,
                    'passed': zscore_margin >= threshold,
                    'zscore_0': sample['zscore_to_class_0'],
                    'zscore_1': sample['zscore_to_class_1']
                }
                
                if zscore_margin >= threshold:
                    zscore_margin_pass_count += 1
            else:
                zscore_margin_info[client] = {
                    'zscore_margin': 0,
                    'threshold': 0,
                    'passed': False,
                    'zscore_0': None,
                    'zscore_1': None
                }
        
        if zscore_margin_pass_count >= min_margin_clients:
            refined_sample = consensus_sample.copy()
            refined_sample['zscore_margin_info'] = zscore_margin_info
            refined_sample['zscore_margin_pass_count'] = zscore_margin_pass_count
            refined_sample['avg_zscore_margin'] = np.mean([info['zscore_margin'] for info in zscore_margin_info.values() if info['zscore_margin'] > 0])
            
            zscore_margin_refined_samples.append(refined_sample)
    
    original_count = len(consensus_samples)
    refined_count = len(zscore_margin_refined_samples)
    
    print(f"\n📈 Z-score Margin Refinement Results:")
    print(f"   Original consensus samples: {original_count}")
    print(f"   Z-score margin-refined samples: {refined_count}")
    print(f"   Refinement rate: {100*refined_count/original_count:.1f}%")
    
    correct_refined = sum(1 for s in zscore_margin_refined_samples if s['is_correct'])
    refined_accuracy = 100 * correct_refined / refined_count if refined_count > 0 else 0
    
    print(f"   Refined accuracy: {correct_refined}/{refined_count} ({refined_accuracy:.1f}%)")

    y_true = [s['true_label'] for s in zscore_margin_refined_samples]
    y_pred = [s['consensus_class'] for s in zscore_margin_refined_samples]

    if y_true:
        macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        cm = confusion_matrix(y_true, y_pred)

        print(f"   Macro F1-score: {macro_f1:.4f}")
        print(f"   Weighted F1-score: {weighted_f1:.4f}")
        print(f"   Confusion Matrix:\n{cm}")
    else:
        print("   No samples to evaluate F1-scores.")
    
    class_counts = defaultdict(int)
    class_correct = defaultdict(int)
    
    for sample in zscore_margin_refined_samples:
        pred_class = sample['consensus_class']
        class_counts[pred_class] += 1
        if sample['is_correct']:
            class_correct[pred_class] += 1
    
    print(f"   Class distribution:")
    for cls in sorted(class_counts.keys()):
        if class_counts[cls] > 0:
            cls_acc = 100 * class_correct[cls] / class_counts[cls]
            print(f"     Class {cls}: {class_counts[cls]} samples ({cls_acc:.1f}% accuracy)")
    
    return zscore_margin_refined_samples, zscore_margin_thresholds


def main():
    parser = argparse.ArgumentParser(description='Prototype Distance-based Collaborative Inference')
    parser.add_argument('--config', type=str)
    parser.add_argument('--device', type=str)
    args = parser.parse_args()
    
    print("=" * 80)
    print("🚀 Starting Prototype Distance-based Collaborative Inference")
    print("=" * 80)
    cfg, device = load_config_and_setup(args.config, args.device)
    
    print("\n📦 Loading data...")
    train_loader, test_loader, combined_loader = load_data(cfg, device)
    
    print("\n🔧 Loading models...")
    pretrained_mae, gp = load_models(cfg, device)
    
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
    
    if cfg['scenario'] == 'unlabeled':
        print("\n🎯 Extracting features and predictions... (Unlabeled Scenario)")
        client_feature_dicts = extract_all_clients_feature_entropy_with_gp(client_models, test_loader, device, gp)
    else:
        print("\n🎯 Extracting features and predictions... (Late-joining Scenario)")
        client_feature_dicts = extract_all_clients_feature_entropy_with_gp(client_models, combined_loader, device, gp)
    
    
    print("\n🔧 Applying Z-score normalization...")
    normalized_client_dicts, norm_stats = apply_zscore_normalization(client_feature_dicts)
    
    print(f"\n🔍 Filtering with Z-score threshold: {cfg['ZETA']}...")
    filtered_client_dicts = filter_by_zscore_threshold(normalized_client_dicts, threshold=cfg['ZETA'])
    
    print("\n🤝 Finding consensus samples...")
    consensus_results = find_consensus_intersection_samples(filtered_client_dicts, min_clients=cfg['KAPPA'])
    
    print(f"\n🎯 Refining with Z-score margin filtering (top {cfg['BETA']}%)...")
    refined_samples, thresholds = refine_with_zscore_margin_filtering(
        consensus_results[cfg['KAPPA']], 
        normalized_client_dicts,
        top_percentage=cfg['BETA'],
        min_margin_clients=cfg['KAPPA']
    )
    
    print("\n" + "=" * 80)
    print("✅ Prototype Distance-based Collaborative Inference Completed!")
    print("=" * 80)
    
    return refined_samples, thresholds, client_feature_dicts, normalized_client_dicts


if __name__ == "__main__":
    refined_samples, thresholds, client_feature_dicts, normalized_client_dicts = main()

