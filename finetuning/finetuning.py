import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
import numpy as np
import pickle

from data.dataset_utils import *
from models.mae import MaskedAutoencoderViT
from models.classifier import *

import argparse
import json
import yaml
import time
from pathlib import Path
from utils import fix_seeds, setup_cudnn

def main(cfg):
    start_time = time.time()
    save_dir = cfg['SAVE_PATH']+cfg['NAME']+'/'+cfg['GOAL']+'/'
    os.makedirs(save_dir, exist_ok=True)
    fix_seeds(cfg['SEED'])
    setup_cudnn()
    device = cfg['DEVICE']
    
    train_transform = transforms.Compose([
                    transforms.RandomResizedCrop(cfg['IMG_SIZE'], scale=(0.6, 1.)),
                    transforms.RandomRotation(degrees=10),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(cfg['MEAN'], cfg['STD'])])

    transform = transforms.Compose([
                transforms.Resize(size=cfg['IMG_SIZE']),
                transforms.CenterCrop(size=(cfg['IMG_SIZE'], cfg['IMG_SIZE'])), 
                transforms.ToTensor(),
                transforms.Normalize(cfg['MEAN'], cfg['STD'])])
    
    print("Loading Dataset...")
    train_dataset = eval(cfg['DATASET'])(cfg['IMG_DIR'], cfg['CSV_DIR'], train_ratio=cfg['TRAIN_RATIO'], train=True, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=cfg['BATCH_SIZE'], shuffle=True, num_workers=4)

    test_dataset = eval(cfg['DATASET'])(cfg['IMG_DIR'], cfg['CSV_DIR'], train_ratio=cfg['TRAIN_RATIO'], train=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=cfg['BATCH_SIZE'], shuffle=False, num_workers=4)
    
    pretrained_mae = MaskedAutoencoderViT(
        img_size=cfg['IMG_SIZE'],
        patch_size=cfg['PATCH_SIZE'],
        hybrid=cfg['HYBRID'] 
    ).to(device)

    if cfg['HYBRID']:
        print("CPE")
    else:
        print("LPE")

    print(f"Freeze: {cfg['FREEZE']}")

    if cfg['SCRATCH']:
        print("Scratch Training")
    else:
        print("Use Pretrained")
        state_dict = torch.load(cfg['MODEL_PATH'], map_location=device, weights_only=True)

        if cfg['BN_PATH'] is not None:
            bn_state_dict = torch.load(cfg['BN_PATH'], map_location=device, weights_only=True)
            state_dict.update(bn_state_dict)
            print("BN")
        else:
            print("AVG")

        if isinstance(state_dict, nn.Module):
            state_dict = state_dict.state_dict()
            print("LOAD")

        pretrained_mae.load_state_dict(state_dict, strict=True)
        pretrained_mae.eval()

    if cfg['ADAPFT']:
        model = FineTunedMAE_Shallow(pretrained_mae, num_classes=cfg['NUM_CLASSES'], freeze=cfg['FREEZE']).to(device)
        print("AdapFT")
    else:
        model = FineTunedMAE(pretrained_mae, num_classes=cfg['NUM_CLASSES'], freeze=cfg['FREEZE']).to(device)
        print("No Adapter")

    
    optimizer = optim.AdamW([
        {"params": model.parameters(), "lr": float(cfg['ENCODER_LR'])}
     ], weight_decay=float(cfg['WEIGHT_DECAY']))
    
    criterion = nn.CrossEntropyLoss()

    metrics = defaultdict(list)
    best_epoch = 0
    best_test_acc = 0.0 
    best_cm = None
    train_loss_list = []
    train_acc_list = []    
    test_acc_list = []
    f1_list = []     

    print("🚀 Training Start 🚀")
    for epoch in range(cfg['EPOCHS']):
        model.train()
        total_loss = 0.0
        train_total_correct = 0
        train_total_samples = 0

        for batch in train_loader:
            x, y = batch
            x, y = x.to(device), y.to(device).long()

            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            predicted = torch.argmax(logits, dim=1)
            train_total_correct += (predicted == y).sum().item()
            train_total_samples += y.size(0)

        avg_loss = total_loss / train_total_samples
        train_accuracy = 100.0 * train_total_correct / train_total_samples
        train_acc_list.append(train_accuracy)
        train_loss_list.append(avg_loss)
        print(f"Epoch: {epoch+1} | Train Loss: {avg_loss:.4f} | Accuracy: {train_accuracy:.2f}%")

        model.eval()
        test_total_correct = 0
        test_total_samples = 0
        all_labels = []
        
        all_preds = []

        with torch.no_grad():
            for batch in test_loader:
                x, y = batch
                x, y = x.to(device), y.to(device).long()

                logits = model(x)
                
                predicted = torch.argmax(logits, dim=1)
                test_total_correct += (predicted == y).sum().item()
                test_total_samples += y.size(0)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        
        test_accuracy = 100.0 * test_total_correct / test_total_samples
        f1 = f1_score(all_labels, all_preds, average='weighted')
        cm = confusion_matrix(all_labels, all_preds)

        print(f"Epoch: {epoch+1} | Test Acc: {test_accuracy:.2f}% | F1: {f1:.4f}")
        # print(f"Confusion Matrix:\n{cm}")
        test_acc_list.append(test_accuracy)   
        f1_list.append(f1)

        if test_accuracy > best_test_acc:
            best_test_acc = test_accuracy
            best_epoch = epoch + 1
            best_cm = cm.tolist() 
            torch.save(model, save_dir + 'best_model.pt')
            print(f"** Best model saved with Accuracy: {best_test_acc:.2f}% at epoch {best_epoch}")

    metrics['Training_Loss'] = train_loss_list
    metrics['Training_Acc'] = train_acc_list
    metrics['Test_Acc'] = test_acc_list
    metrics['f1'] = f1_list
    
    metrics_path = save_dir + 'metrics.json'
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
        
    if best_cm is not None:
        with open(save_dir + 'confusion_matrix.json', 'w') as f:
            json.dump({"Best_Epoch": best_epoch, "Confusion_Matrix": best_cm}, f, indent=4)


    cfg_path = save_dir + 'config.yaml'
    with open(cfg_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
            
    training_time = time.time() - start_time
    print(f"Total training time: {training_time:.2f}s")
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='./configs/finetuning/nmc.yaml')
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.safe_load(f)

    main(cfg)
