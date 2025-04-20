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
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split


#Seccion para la division entre train y test de los archivos de audio
df = pd.read_csv(r"C:\Users\esteb\OneDrive\Desktop\Proyectos programacion\PI II\data\outputA\resultados.csv")
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

train_df.to_csv("train.csv", index=False)
test_df.to_csv("test.csv", index=False)

#Datos fijos predefinidos para el manejo de transformaciones de las waveform al espectrograma
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

#Funcion que recolecta los paths validos de archivos de audio
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

#data = collect_valid_flac_paths(r"C:\Users\esteb\OneDrive\Desktop\Proyectos programacion\PI II\ASVspoof2021_LA_eval\ASVspoof2021_LA_eval\flac", limit=40000)

#Codigo para clase creada que maneja Espectrogramas Mel
class MelSpectrogramDataset(Dataset):
    
    
    def __init__(self, csv_path, audios="audio", duration=5.0, sr=16000, augment=True):
        self.data = pd.read_csv(csv_path)
        self.audio_paths = self.data[audios].tolist()
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
        return len(self.audio_paths)

    def __getitem__(self, idx):
        path = self.audio_paths[idx]
        
        try:
            y, sr = librosa.load(path, sr=16000)
            waveform = torch.tensor(y).unsqueeze(0)
           
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
                waveform = self.augmentation(waveform)
               
            #Normalización
            waveform = waveform / waveform.abs().max()
           
            mel = self.mel_transform(waveform)
           
            mel_db = self.db_transform(mel)
          
            mel_db = self.standardize(mel_db)
          
            return mel_db  
        
        except Exception as e:
            print(f"[ERROR cargando {path}] {e}")
            return torch.zeros(1, 64, int(self.sr * self.duration / 160)) 

    #Funcion para aplicar transformaciones de data augmentation al waveform
    def augmentation(self, waveform):
        augmentations = [self.add_noise, self.change_volume]
        aug = random.choice(augmentations)
        waveform = aug(waveform)
        return waveform
    
    #Funcion de transformacion de añadir ruido background        
    def add_noise(self, waveform, noise_level=0.005):
        noise = torch.randn_like(waveform)
        return waveform + noise_level * noise

    #Funcion de estandarizacion
    def standardize(self, spec):
        return (spec - spec.mean()) / (spec.std() + 1e-9)
    
    #Fucnion de transformación cambiando el volumen del audio
    def change_volume(self, waveform, gain_db=5.0):
        factor = 10 ** (gain_db / 20)
        return waveform * factor
    
    def apply_codec(self, waveform, sample_rate, format, encoder=None):
        encoder = torchaudio.io.AudioEffector(format=format, encoder=encoder)
        return encoder.apply(waveform, sample_rate)
    
    #Generador de grafica del espectrograma antes de la estandarizacion
    def before_change(self, mel):
        plt.imshow(mel.squeeze().numpy(), origin='lower', aspect='auto')
        plt.title("Espectrograma")
        plt.colorbar()
        plt.xlabel("Tiempo")
        plt.ylabel("Frecuencia Mel")
        plt.tight_layout()
        plt.show()

#Instanciacion de los dataset train y test
train_dataset = MelSpectrogramDataset(csv_path="train.csv", augment=True)
test_dataset = MelSpectrogramDataset(csv_path="test.csv", augment=True)

subset = torch.utils.data.Subset(train_dataset, list(range(5000)))

mel = subset[3] 

#Generador de grafica del espectrograma despues de la estandarizacion
plt.imshow(mel.squeeze().numpy(), origin='lower', aspect='auto')
plt.title("Espectrograma")
plt.colorbar()
plt.xlabel("Tiempo")
plt.ylabel("Frecuencia Mel")
plt.tight_layout()
plt.show()

#Impresion de datos importantes generales para analisis promedio de los audios
print(f"Min: {mel.min().item():.2f}")
print(f"Max: {mel.max().item():.2f}")
print(f"Mean: {mel.mean().item():.2f}")
print(f"Std:  {mel.std().item():.2f}")

#Instanciacion de los dataloaders train y test
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

# Ejemplo iterando
for audio in train_dataloader:
    print(audio.shape)  # (32, 3, 224, 224)
    break

for audio in test_dataloader:
    print(audio.shape)  # (32, 3, 224, 224)
    break




