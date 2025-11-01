"""
Utility functions for the project
"""

import os
import random
import numpy as np
import torch
import yaml
from pathlib import Path
from typing import Dict, Any


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_directories(config: Dict[str, Any]):
    """
    Create necessary directories for the project

    Args:
        config: Configuration dictionary
    """
    paths = config.get('paths', {})
    for key, path in paths.items():
        Path(path).mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    Path(paths.get('figures', 'results/figures')
         ).mkdir(parents=True, exist_ok=True)
    Path(paths.get('metrics', 'results/metrics')
         ).mkdir(parents=True, exist_ok=True)
    Path(paths.get('logs', 'results/logs')).mkdir(parents=True, exist_ok=True)
    Path(paths.get('checkpoints', 'models/checkpoints')
         ).mkdir(parents=True, exist_ok=True)


def get_device(config: Dict[str, Any]) -> torch.device:
    """
    Get device for training (CPU, CUDA GPU, or MPS for Apple Silicon)

    Args:
        config: Configuration dictionary

    Returns:
        PyTorch device
    """
    device_config = config.get('device', {})
    use_cuda = device_config.get('use_cuda', True)
    gpu_id = device_config.get('gpu_id', 0)

    # Try CUDA first (NVIDIA GPUs)
    if use_cuda and torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu_id}')
        print(f"Using CUDA GPU: {torch.cuda.get_device_name(gpu_id)}")
    # Try MPS (Apple Silicon)
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device('mps')
        print("Using MPS (Apple Silicon GPU)")
    # Fallback to CPU
    else:
        device = torch.device('cpu')
        print("Using CPU")
        if use_cuda:
            print("  Note: CUDA was requested but is not available")

    return device


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count trainable parameters in a model

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    path: str
):
    """
    Save model checkpoint

    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        metrics: Dictionary of metrics
        path: Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    path: str,
    device: torch.device
) -> Dict[str, Any]:
    """
    Load model checkpoint

    Args:
        model: PyTorch model
        optimizer: Optimizer
        path: Path to checkpoint
        device: Device to load model to

    Returns:
        Checkpoint dictionary
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Checkpoint loaded from {path}")
    return checkpoint
