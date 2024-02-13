"""
Dataset and dataloader for MNIST dataset using flat vector representations
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from typing import Optional, Callable
from pathlib import Path

class MNISTDataset(Dataset):
    """Download the full MNIST dataset (train subset) and provide flat vector representations"""
    def __init__(self, data_dir: str= Path(os.getcwd(),'/data/'), 
                 train: bool=True, 
                 transform: Optional[Callable] = None, 
                 target_transform: Optional[Callable] = None, 
                 download: bool=True):
        self.mnist = datasets.MNIST(root=data_dir, train=train, transform=transform, target_transform=target_transform, download=download)
        self.x = self.mnist.data
        self.y = self.mnist.targets
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        img, target = self.x[idx], self.y[idx]
        
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target