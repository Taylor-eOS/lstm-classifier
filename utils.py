import os
import torch
from torch.utils.data import Dataset
import numpy as np
import librosa

def preprocess_audio(file_path, sampling_rate=8000, n_mels=40, seq_length=100):
    # Load and resample audio
    audio, sr = librosa.load(file_path, sr=sampling_rate)
    # Extract Mel spectrogram
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sampling_rate, n_mels=n_mels)
    # Convert to log scale
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    # Transpose to have time steps in rows
    log_mel_spec = log_mel_spec.T
    # Pad or truncate to fixed sequence length
    if log_mel_spec.shape[0] < seq_length:
        pad_width = seq_length - log_mel_spec.shape[0]
        log_mel_spec = np.pad(log_mel_spec, ((0, pad_width), (0, 0)), mode='constant')
    else:
        log_mel_spec = log_mel_spec[:seq_length, :]
    return log_mel_spec

class AudioDataset(Dataset):
    def __init__(self, data_dir, seq_length=100, sampling_rate=8000, n_mels=40):
        self.data = []
        self.labels = []
        self.seq_length = seq_length
        self.sampling_rate = sampling_rate
        self.n_mels = n_mels

        # Assuming two subdirectories: 'ads' and 'non-ads'
        classes = ['ads', 'non-ads']
        for label, class_name in enumerate(classes):
            class_dir = os.path.join(data_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for file_name in os.listdir(class_dir):
                if file_name.endswith('.wav'):
                    file_path = os.path.join(class_dir, file_name)
                    self.data.append(file_path)
                    self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_path = self.data[idx]
        label = self.labels[idx]
        features = preprocess_audio(
            file_path, sampling_rate=self.sampling_rate, n_mels=self.n_mels, seq_length=self.seq_length)
        features = torch.tensor(features, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        return features, label

