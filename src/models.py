"""
CNN Model Architectures - Models 1-3
Part of 10-model comparison for Animals-10 classification

Each model follows the brief requirements:
- Convolutional layers for feature extraction
- Pooling layers for downsampling
- Fully connected layers for classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model1_BaseCNN(nn.Module):
    """
    Model 1: Base CNN (No Regularization)
    
    Architecture:
    - Max pooling after each conv layer
    - 2 Fully connected layers
    - No dropout, no batch norm
    
    Purpose: Baseline model to compare all other models against.
    Expected: Overfitting (high train acc, lower val acc)
    """
    
    def __init__(self, num_classes=10):
        super(Model1_BaseCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 28 * 28, 256)
        self.fc2 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x