import copy
import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import DataLoader
from urllib.request import urlopen
from PIL import Image
import timm
import torch.nn.functional as F
from collections import defaultdict
from tqdm import tqdm
from models.mae import MaskedAutoencoderViT
from data.dataset_utils import *
import time
import torchvision.transforms as transforms
from torch.cuda.amp import autocast, GradScaler
import random
import torch.backends.cudnn as cudnn

class Client(object):
    def __init__(self, cfg, id, dataset, img_path, csv_path, **kwargs):
        self.dataset = dataset
        self.dataset_name = cfg['CLIENTS'][self.dataset]['NAME']
        self.device = cfg['COMMON']['DEVICE']
        self.img_size = cfg['COMMON']['IMAGE_SIZE']
        self.patch_size = cfg['COMMON']['PATCH_SIZE']
        self.mean = cfg['COMMON']['MEAN']
        self.std = cfg['COMMON']['STD']
        self.local_epochs = cfg['CLIENTS'][self.dataset]['LOCAL_EPOCH']
        self.batch_size = cfg['CLIENTS'][self.dataset]['BATCH_SIZE']
        self.mask_ratio = cfg['CLIENTS'][self.dataset]['MASK_RATIO']
        self.train_ratio = cfg['CLIENTS'][self.dataset]['TRAIN_RATIO']
        self.betas = cfg['CLIENTS'][self.dataset]['BETAS']
        self.weight_decay = cfg['CLIENTS'][self.dataset]['WEIGHT_DECAY']
        self.eps = float(cfg['CLIENTS'][self.dataset]['EPS'])
        self.accum_iter = cfg['CLIENTS'][self.dataset].get('ACCUM_ITER', 1)
        self.base_lr = float(cfg['COMMON']['BASE_LR'])
        self.warmup_epochs = cfg['COMMON']['WARMUP_EPOCHS']
        self.max_rounds = cfg['SERVER']['GLOBAL_ROUNDS']

        self.model = eval(cfg['COMMON']['MODEL'])(
            img_size=self.img_size,
            patch_size=self.patch_size,
            hybrid=cfg['COMMON']['HYBRID']
        ).to(self.device)



        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=1e-8,  # placeholder, will be overridden by scheduler
            betas=self.betas,
            weight_decay=self.weight_decay,
            eps=self.eps
        )

        self.scaler = GradScaler()

        self.img_path = img_path
        self.csv_path = csv_path

        self.transform = transforms.Compose([
                    transforms.RandomResizedCrop(self.img_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic   
                    transforms.RandomGrayscale(p=0.2),
                    transforms.ColorJitter(0.4, 0.4, 0.4),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std)])

        self.id = id

        self.global_step = 0
        self.total_steps = self.max_rounds * self.local_epochs  # approximate
        self.warmup_steps = self.warmup_epochs * self.local_epochs

        self.train_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
        self.send_time_cost = {'num_rounds': 0, 'total_cost': 0.0}

    def set_lr(self, lr):
        for g in self.optimizer.param_groups:
            g['lr'] = lr

    def adjust_learning_rate(self, progress):
        if progress < self.warmup_epochs:
            lr = self.base_lr * progress / self.warmup_epochs
        else:
            decay_progress = (progress - self.warmup_epochs) / (self.max_rounds - self.warmup_epochs)
            lr = self.base_lr * 0.5 * (1.0 + np.cos(np.pi * decay_progress))
        self.set_lr(lr)

    def load_local_data(self):
        dataset = eval(self.dataset_name)(self.img_path, self.csv_path, train_ratio=self.train_ratio, train=True, transform=self.transform)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def set_parameters(self, model):
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()

    def train(self, global_epoch):
        train_loader = self.load_local_data()
        self.train_samples = len(train_loader.dataset)
        start_time = time.time()
        self.model.train()

        self.optimizer.zero_grad()
        step_count = 0
        total_loss = 0

        for epoch in range(self.local_epochs):
            total_batches = len(train_loader)

            for batch_idx, (x, _) in enumerate(train_loader):
                x = x.to(self.device)

                progress = global_epoch + batch_idx / total_batches
                self.adjust_learning_rate(progress)

                with autocast(enabled=True):
                    loss, pred, mask = self.model(x, mask_ratio=self.mask_ratio)
                    loss = loss / self.accum_iter

                self.scaler.scale(loss).backward()

                if (batch_idx + 1) % self.accum_iter == 0 or (batch_idx + 1) == len(train_loader):
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()

                total_loss += loss.item() * self.accum_iter
                step_count += 1

        avg_loss = total_loss / (len(train_loader) * self.local_epochs)

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

        return avg_loss

    def train_metrics(self):
        train_loader = self.load_local_data()
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in train_loader:
                x, _ = batch
                x = x.to(self.device)
                loss, pred, mask = self.model(x, mask_ratio=self.mask_ratio)
                total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        return avg_loss
