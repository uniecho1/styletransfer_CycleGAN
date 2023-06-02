import torch


def try_gpu(i=0):
    """尝试使用 GPU"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')
