import os
import librosa
import torchaudio
import torch.nn as nn
import torch
import numpy as np
import torchaudio.transforms as T
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pandas as pd
import glob
import random

class SpectrogramCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)         # (B, 1, 64, T) → (B, 6, 60, T-4)
        self.pool = nn.MaxPool2d(2, 2)                      # Mitad: (B, 6, 30, (T-4)//2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)        # (B, 16, 26, ...)
        
        
        self.fc1 = nn.Linear(16 * 13 * 122, 120)
        self.fc2 = nn.Linear(120, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Conv1 → ReLU → Pool
        x = self.pool(F.relu(self.conv2(x)))  # Conv2 → ReLU → Pool
        x = torch.flatten(x, 1)               # Aplanar todo excepto batch
        x = F.relu(self.fc1(x))               # Linear → ReLU
        x = self.fc2(x)                       # Output
        return x