import torch
import numpy as np
import random
import time
import os
import functools
from pathlib import Path
from torch.backends import cudnn
from torch import nn, Tensor
from torch.autograd import profiler
from typing import Union
from torch import distributed as dist
import pickle
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms

def load_public_indices(base_dir):
    load_path = os.path.join(base_dir, f"public_indices.pkl")
    
    with open(load_path, 'rb') as f:
        indices = pickle.load(f)
    print(f"Public indices loaded from {load_path}")
    return indices

def load_client_indices(base_dir, dataset_type, client_id):
    """
    Load saved client indices from disk.

    Args:
        base_dir (str): Base directory path
        dataset_type (str): Dataset type ("train", "val", "test")
        client_id (int): Client ID

    Returns:
        list: Loaded index list
    """
    load_path = os.path.join(base_dir, f"{dataset_type}_indices", f"client_{client_id}_indices.pkl")
    
    with open(load_path, 'rb') as f:
        indices = pickle.load(f)
    print(f"Client {client_id} {dataset_type} indices loaded from {load_path}")
    
    return indices

def create_dataloader_from_indices(dataset, indices, batch_size=32, shuffle=True):
    """
    Create a DataLoader with only the data at given indices.

    Args:
        trainset: Original dataset
        indices (list): List of indices to use
        batch_size (int): Batch size
        shuffle (bool): Whether to shuffle data

    Returns:
        DataLoader: Created dataloader
    """
    subset = Subset(dataset, indices)
    
    dataloader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
    )
    
    print(f"Created DataLoader with {len(subset)} samples")
    print(f"Number of batches: {len(dataloader)}")
    
    return dataloader

def fix_seeds(seed: int = 3407) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def setup_cudnn() -> None:
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    cudnn.benchmark = True
    cudnn.deterministic = False

def time_sync() -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()

def get_model_size(model: Union[nn.Module, torch.jit.ScriptModule]):
    tmp_model_path = Path('temp.p')
    if isinstance(model, torch.jit.ScriptModule):
        torch.jit.save(model, tmp_model_path)
    else:
        torch.save(model.state_dict(), tmp_model_path)
    size = tmp_model_path.stat().st_size
    os.remove(tmp_model_path)
    return size / 1e6   # in MB

@torch.no_grad()
def test_model_latency(model: nn.Module, inputs: torch.Tensor, use_cuda: bool = False) -> float:
    with profiler.profile(use_cuda=use_cuda) as prof:
        _ = model(inputs)
    return prof.self_cpu_time_total / 1000  # ms

def count_parameters(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6      # in M

def setup_ddp() -> int:
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        gpu = int(os.environ(['LOCAL_RANK']))
        torch.cuda.set_device(gpu)
        dist.init_process_group('nccl', init_method="env://",world_size=world_size, rank=rank)
        dist.barrier()
    else:
        gpu = 0
    return gpu

def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()

def reduce_tensor(tensor: Tensor) -> Tensor:
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt

@torch.no_grad()
def throughput(dataloader, model: nn.Module, times: int = 30):
    model.eval()
    images, _  = next(iter(dataloader))
    images = images.cuda(non_blocking=True)
    B = images.shape[0]
    print(f"Throughput averaged with {times} times")
    start = time_sync()
    for _ in range(times):
        model(images)
    end = time_sync()

    print(f"Batch Size {B} throughput {times * B / (end - start)} images/s")


def timer(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        tic = time.perf_counter()
        value = func(*args, **kwargs)
        toc = time.perf_counter()
        elapsed_time = toc - tic
        print(f"Elapsed time: {elapsed_time * 1000:.2f}ms")
        return value
    return wrapper_timer

