import torch 
import argparse
import yaml
import time
from pathlib import Path
from serverproto import FedProto
from serveravg import FedAvg
from serverrep import FedRep
from utils import fix_seeds, setup_cudnn
import os
import pickle

def main(cfg):
    start_time = time.time()
    fix_seeds(cfg['COMMON']['SEED'])
    setup_cudnn()

    if cfg['SERVER']['ALGORITHM'] == 'fedavg':
        server = FedAvg(cfg=cfg, times=start_time)
        print('Run FedAvg')
    elif cfg['SERVER']['ALGORITHM'] == 'fedrep':
        server = FedRep(cfg=cfg, times=start_time)
        print('Run FedRep')
    elif cfg['SERVER']['ALGORITHM'] == 'fedproto':
        server = FedProto(cfg=cfg, times=start_time)
        print('Run FedProto')
    else:
        raise NotImplementedError

    print("🚀 Training Start 🚀")
    server.train()
    training_time = time.time() - start_time
    print(f"Total training time: {training_time:.2f}s")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='./configs/common.yaml')
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.safe_load(f)

    main(cfg)

