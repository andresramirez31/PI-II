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

class SpectrogramCNNMulticlass(nn.Module):
    
    def __init__(self, num_etiquetas, num_nombres):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.flattened_size = 16 * 13 * 122  # ajusta según tu input real
        self.fc = nn.Linear(self.flattened_size, 120)

        # 2 salidas: una por tarea
        self.fc_etiqueta = nn.Linear(120, num_etiquetas)
        self.fc_nombre = nn.Linear(120, num_nombres)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc(x))

        out_etiqueta = self.fc_etiqueta(x)
        out_nombre = self.fc_nombre(x)
        return out_etiqueta, out_nombre