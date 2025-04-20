import os
import torchaudio
import torch
import librosa
import numpy as np
import torchaudio.transforms as T
import matplotlib.pyplot as plt
from pathlib import Path

PTT = 400 
SAMPLE_RATE = 16000
WIN_LENGTH = 400
HOP_LENGTH = 200
N_FFT = 512
N_MELS = 65
DURATION = 4 

durations = []
sample_rates = []
espectrogramas = []
i = 0


def preprocesamiento_a_mel(path):
    waveform, sr = torchaudio.load(path)
    
    
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    # Resampleado
    if sr != SAMPLE_RATE:
        resampler = T.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
        waveform = resampler(waveform)
        
    # Ajuste para longitud Fija
    longitud_esperada = int(SAMPLE_RATE * DURATION)
    if waveform.shape[1] > longitud_esperada:
        waveform = waveform[:, :longitud_esperada]
    else:
        padding = longitud_esperada - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, padding))
    
    mel_transform = T.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=WIN_LENGTH,         # ~25ms window
    hop_length=HOP_LENGTH,    # ~10ms stride
    n_mels=N_MELS
    )

    db_transform = T.AmplitudeToDB(top_db=80)
    
    
    
    mel_spec = mel_transform(waveform)
    mel_db = db_transform(mel_spec)
    
    mel_db =  (mel_db - mel_db.mean()) / (mel_db.std() + 1e-9)

    
    return mel_db


for root, _, files in os.walk(r"C:\Users\esteb\OneDrive\Desktop\Proyectos programacion\PI II\ASVspoof2021_LA_eval\ASVspoof2021_LA_eval\flac"):
    for file in files:
        if file.endswith(".flac") and i < 100:
            path = os.path.join(root, file)
            try:
                waveform, sr = torchaudio.load(path)
                duration = waveform.shape[1] / sr
                sample_rates.append(sr)
                durations.append(duration)
                print(duration)
                mel = preprocesamiento_a_mel(path)
                espectrogramas.append(mel)
                
                i = i + 1
            except Exception as e:
                print(f"❌ Failed to load {path} | Reason: {e}")
                continue

plt.imshow(mel[0].numpy(), origin='lower', aspect='auto')
plt.title('Espectrograma Mel')
plt.colorbar()
plt.show()

print(f"Unique sample rates: {set(sample_rates)}")
print(f"Min duration: {np.min(durations):.2f}s")
print(f"Max duration: {np.max(durations):.2f}s")
print(f"Average duration: {np.mean(durations):.2f}s")


waveform, sr = torchaudio.load(r"C:\Users\esteb\OneDrive\Desktop\Proyectos programacion\PI II\ASVspoof2021_LA_eval\ASVspoof2021_LA_eval\flac\LA_E_1000200.flac")
print("done")

#Progreso 7 abril