from torch.utils.data import Dataset
import librosa
import torchaudio
import torch
import matplotlib.pyplot as plt
import pandas as pd
import random
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split



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


#Codigo para clase creada que maneja Espectrogramas Mel
class MelSpectrogramDataset(Dataset):
    
    
    def __init__(self, csv_path, audios="audio", duration=5.0, label_columns=None, sr=16000, augment=True):
        self.data = pd.read_csv(csv_path)
        self.audio_paths = self.data[audios].tolist()
        self.duration = duration
        self.label_columns = label_columns if label_columns else []
        self.labels = self.data[self.label_columns] if self.label_columns else None
        self.sr = sr
        self.augment = augment
        
        self.encoders = {}
        if self.label_columns:
            for col in self.label_columns:
                encoder = LabelEncoder()
                self.data[col] = encoder.fit_transform(self.data[col].astype(str))
                self.encoders[col] = encoder
            self.labels = self.data[self.label_columns].astype('int64')
        else:
            self.labels = None

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
               
            label = self.labels.iloc[idx].values if self.labels is not None else None
            #Normalización
            waveform = waveform / waveform.abs().max()
           
            mel = self.mel_transform(waveform)
           
            mel_db = self.db_transform(mel)
          
            mel_db = self.standardize(mel_db)
          
            if self.labels is not None:
                label = self.labels.iloc[idx].values.astype(int)  # Garantizamos que son int
                label = torch.tensor(label, dtype=torch.long)
                
            else:
                label = torch.tensor([-1])
                
            return mel_db, label 
        
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

def train_test_division(resultados_path):
    #Seccion para la division entre train y test de los archivos de audio
    df = pd.read_csv(resultados_path)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    train_df.to_csv("train.csv", index=False)
    test_df.to_csv("test.csv", index=False)


def dataloaders(train_csv="train.csv", test_csv="test.csv", batch_size=32):
#Instanciacion de los dataset train y test
    train_dataset = MelSpectrogramDataset(csv_path=train_csv, label_columns=["etiqueta", "nombre"], augment=True)
    test_dataset = MelSpectrogramDataset(csv_path=test_csv, label_columns=["etiqueta", "nombre"], augment=True)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_dataloader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    return train_dataloader, test_dataloader, train_dataset, test_dataset


def main():
    
    train_test_division(r"data/outputA/resultados.csv")
    
    train_loader, _, train_dataset, _ = dataloaders()
    subset = torch.utils.data.Subset(train_dataset, list(range(5000)))

    mel, label = subset[3] 

    print(label)
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


    # Ejemplo iterando
    for audio, label in train_loader:
        print(audio.shape)  # (32, 3, 224, 224)
        print(label.shape)
        break
    
if __name__ == "__main__":
    
    main()
     






