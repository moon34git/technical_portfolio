import torch 
import argparse
import yaml
import time
from pathlib import Path
from server import Server
from utils import fix_seeds, setup_cudnn
import os
import pickle

def main(cfg):
    start_time = time.time()
    fix_seeds(cfg['COMMON']['SEED'])
    setup_cudnn()

    server = Server(cfg=cfg, times=start_time)

    print("🚀 Training Start 🚀")
    server.train()
    training_time = time.time() - start_time
    print(f"Total training time: {training_time:.2f}s")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='./configs/pretraining/custom.yaml')
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.safe_load(f)

    main(cfg)

