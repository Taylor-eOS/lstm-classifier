import os
import shutil
import librosa
import numpy as np
from pydub import AudioSegment
import torch
from torch.utils.data import Dataset

class AudioDataset(Dataset):
    def __init__(self, data_dir, seq_length=100, sampling_rate=4000, n_mfcc=10):
        self.data = []
        self.labels = []
        self.seq_length = seq_length
        self.sampling_rate = sampling_rate
        self.n_mfcc = n_mfcc
        classes = ['A', 'B']
        for label, class_name in enumerate(classes):
            class_dir = os.path.join(data_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for file_name in os.listdir(class_dir):
                if file_name.lower().endswith('.wav'):
                    file_path = os.path.join(class_dir, file_name)
                    self.data.append(file_path)
                    self.labels.append(label)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        file_path = self.data[idx]
        label = self.labels[idx]
        features = preprocess_audio(file_path, sampling_rate=self.sampling_rate, n_mfcc=self.n_mfcc, seq_length=self.seq_length)
        features = torch.tensor(features, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        return features, label

def preprocess_audio(file_path, sampling_rate=4000, n_mfcc=10, seq_length=100, n_fft=512, hop_length=256):
    audio, sr = librosa.load(file_path, sr=sampling_rate)
    #print(f"Sampling rate: {sr}, seqence length: {seq_length}")
    mfcc = librosa.feature.mfcc(y=audio, sr=sampling_rate, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    if mfcc.shape[1] < 5:
        pad_width = 5 - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
    delta_mfcc = librosa.feature.delta(mfcc, width=5)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2, width=5)
    feature = np.concatenate((mfcc, delta_mfcc, delta2_mfcc), axis=0)
    feature = feature.T
    def print_debug(string):
        #print(f'{string} {feature.shape[0]} {seq_length}')
        pass
    if feature.shape[0] < seq_length:
        print_debug('padding')
        pad_width = seq_length - feature.shape[0]
        repeat_frames = np.tile(feature, (pad_width // feature.shape[0] + 1, 1))[:pad_width, :]
        feature = np.concatenate([feature, repeat_frames], axis=0) #Repeat frames starting from the beginning to fill the gap
    elif feature.shape[0] > seq_length:
        print_debug('truncating')
        feature = feature[:seq_length, :]
    else:
        print_debug('using as is')
    return feature

def convert_mp3(mp3_path, wav_path):
    print(f"Converting {os.path.basename(mp3_path)}")
    audio = AudioSegment.from_mp3(mp3_path)
    audio.export(wav_path, format="wav")

def convert_wav(wav_path, mp3_path):
    print(f"Converting {os.path.basename(wav_path)}")
    audio = AudioSegment.from_wav(wav_path)
    audio.export(mp3_path, format="mp3")

def convert_time_to_seconds(time_str):
    parts = time_str.split(':')
    if len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + int(seconds)
    elif len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + int(seconds)
    else:
        raise ValueError(f"Invalid time format: {time_str}")

def create_empty_folder(folder_name):
            if os.path.exists(folder_name):
                shutil.rmtree(folder_name)
            os.makedirs(folder_name, exist_ok=True)

