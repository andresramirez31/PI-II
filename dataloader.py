from torch.utils.data import Dataset
import os
import librosa
import torchaudio
import torch
import numpy as np
import torchaudio.transforms as T
import matplotlib.pyplot as plt
import pandas as pd
import glob
import random





PTT = 400 
SAMPLE_RATE = 16000
WIN_LENGTH = 400
HOP_LENGTH = 200
N_FFT = 512
N_MELS = 65
DURATION = 2 

durations = []
sample_rates = []
espectrogramas = []
i = 0

def collect_valid_flac_paths(root_dir, limit=None):
    good_paths = []
    bad_paths = []
    i = 0

    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".flac"):
                path = os.path.join(root, file)
                try:
                    torchaudio.load(path)
                    good_paths.append(path)
                   
                    i += 1
                    if limit and i >= limit:
                        return good_paths
                except Exception as e:
                    bad_paths.append((path, str(e)))
    return good_paths

data = collect_valid_flac_paths(r"C:\Users\esteb\OneDrive\Desktop\Proyectos programacion\PI II\ASVspoof2021_LA_eval\ASVspoof2021_LA_eval\flac", limit=50)
class MelSpectrogramDataset(Dataset):
    
    def __init__(self, data, duration=5.0, sr=16000, augment=True):
        self.data = data
        self.duration = duration
        self.sr = sr
        self.augment = augment

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB(top_db=80)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path = self.data[idx]
        
        try:
            waveform, sr = torchaudio.load(path)
           
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
                

            if sr != self.sr:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sr)
                waveform = resampler(waveform)
                

            longitud_esperada = int(self.sr * self.duration)
          
            
            if waveform.shape[1] > longitud_esperada:
                waveform = waveform[:, :longitud_esperada]
               
            else:
                padding = longitud_esperada - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, padding))
               

            if self.augment:
                waveform = self.add_noise(waveform)
               

            mel = self.mel_transform(waveform)
           
            mel_db = self.db_transform(mel)
          
            mel_db = self.standardize(mel_db)
          
            return mel_db  
        
        except Exception as e:
            print(f"[ERROR cargando {path}] {e}")
            return torch.zeros(1, 64, int(self.sr * self.duration / 160)) 

    def augmentation(self, waveform):
        augmentations = [self.add_noise, self.vorbis_effect]
        aug = random.choice(augmentations)
        waveform = aug(waveform)
            
    def add_noise(self, waveform, noise_level=0.005):
        noise = torch.randn_like(waveform)
        return waveform + noise_level * noise

    def standardize(self, spec):
        return (spec - spec.mean()) / (spec.std() + 1e-9)
    
    def vorbis_effect(self, waveform):
        waveform = self.apply_codec(waveform, self.sr, "ogg", encoder="vorbis")
        return waveform
    
    def apply_codec(self, waveform, sample_rate, format, encoder=None):
        encoder = torchaudio.io.AudioEffector(format=format, encoder=encoder)
        return encoder.apply(waveform, sample_rate)


dataset = MelSpectrogramDataset(data[:10], augment=True)

subset = torch.utils.data.Subset(dataset, list(range(5)))

mel = subset[0] 

plt.imshow(mel.squeeze().numpy(), origin='lower', aspect='auto')
plt.title("Espectrograma")
plt.colorbar()
plt.xlabel("Tiempo")
plt.ylabel("Frecuencia Mel")
plt.tight_layout()
plt.show()

print(f"Min: {mel.min().item():.2f}")
print(f"Max: {mel.max().item():.2f}")
print(f"Mean: {mel.mean().item():.2f}")
print(f"Std:  {mel.std().item():.2f}")




