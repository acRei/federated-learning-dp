"""
model.py – CNN architecture for federated learning.

A compact convolutional network for MNIST that balances accuracy
with communication efficiency (small parameter count).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MNISTNet(nn.Module):
    """
    Lightweight CNN for MNIST classification.
    
    Architecture:
        Conv2d(1, 16, 5) → ReLU → MaxPool
        Conv2d(16, 32, 5) → ReLU → MaxPool
        Linear(512, 128) → ReLU → Dropout
        Linear(128, 10)
    
    Total params: ~52K (small footprint for efficient federated communication)
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))    # (B, 16, 14, 14)
        x = self.pool(F.relu(self.conv2(x)))    # (B, 32, 7, 7)
        x = x.view(x.size(0), -1)               # (B, 1568)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def get_params(self):
        """Return model parameters as a flat dictionary of tensors."""
        return {name: param.clone() for name, param in self.named_parameters()}

    def set_params(self, params_dict):
        """Load parameters from a flat dictionary."""
        with torch.no_grad():
            for name, param in self.named_parameters():
                if name in params_dict:
                    param.copy_(params_dict[name])

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
