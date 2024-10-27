import os
import time
import shutil
import librosa
import numpy as np
from pydub import AudioSegment
import torch
from torch.utils.data import Dataset

CLASSES = ['A', 'B']
LAST_PRINT_TIME = 0
PRINT_FEATURE_SHAPE = False
PRINT_SAMPLING_RATE = False

class AudioDataset(Dataset):
    def __init__(self, data_dir, seq_length=100, sampling_rate=4000, n_mfcc=10):
        self.data = []
        self.labels = []
        self.seq_length = seq_length
        self.sampling_rate = sampling_rate
        self.n_mfcc = n_mfcc
        for label, class_name in enumerate(CLASSES):
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

def preprocess_audio(file_path, sampling_rate=4000, n_mfcc=10, seq_length=100, hop_length=256, n_fft=512):
    audio, sr = librosa.load(file_path, sr=sampling_rate)
    if PRINT_SAMPLING_RATE: print(f"Sampling rate: {sr}, seqence length: {seq_length}")
    mfcc = librosa.feature.mfcc(y=audio, sr=sampling_rate, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    if mfcc.shape[1] < 5:
        pad_width = 5 - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
    delta_mfcc = librosa.feature.delta(mfcc, width=5)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2, width=5)
    feature = np.concatenate((mfcc, delta_mfcc, delta2_mfcc), axis=0)
    feature = feature.T
    feature_shape = feature.shape[0]
    def print_debug(string):
        global LAST_PRINT_TIME
        current_time = time.time()
        if current_time - LAST_PRINT_TIME >= 30 and PRINT_FEATURE_SHAPE:
            print(f'feature shape: {feature_shape}, sequence length: {seq_length}, {string} {feature_shape - seq_length}')
            LAST_PRINT_TIME = current_time
    if feature_shape < seq_length:
        print_debug('padding')
        pad_width = seq_length - feature_shape
        repeat_frames = np.tile(feature, (pad_width // feature_shape + 1, 1))[:pad_width, :]
        feature = np.concatenate([feature, repeat_frames], axis=0) #Repeat frames from the beginning
    elif feature_shape > seq_length:
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

