"""
Dataset and dataloader for MNIST dataset using flat vector representations
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

class MNISTDataset(Dataset):
    """Download the full MNIST dataset and provide flat vector representations"""
    def __init__(self, data_dir, train=True, transform=None, target_transform=None, download=True):
        self.mnist = datasets.MNIST(data_dir, train=train, transform=transform, target_transform=target_transform, download=download)
        self.data = self.mnist.data
        self.targets = self.mnist.targets
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        img, target = self.data[idx], int(self.targets[idx])

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img.view(-1), target