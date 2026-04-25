"""Configuration values for self-pruning CIFAR-10 training."""

from dataclasses import dataclass, field
from typing import List

import torch


@dataclass
class Config:
    """Central training configuration."""

    data_dir: str = "./data"
    output_dir: str = "./outputs"
    batch_size: int = 128
    learning_rate: float = 1e-3
    epochs: int = 3
    weight_decay: float = 1e-5
    lambda_values: List[float] = field(default_factory=lambda: [1e-4, 1e-3, 1e-2])
    val_split: float = 0.1
    seed: int = 42
    num_workers: int = 2
    hidden_dims: tuple = (1024, 512)
    fake_train_size: int = 5000
    fake_test_size: int = 1000
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
