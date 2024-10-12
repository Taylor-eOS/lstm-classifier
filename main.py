import os
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from torch.utils.data import DataLoader
from utils import AudioDataset, preprocess_audio
import librosa
import numpy as np

# Configuration values
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAMPLING_RATE = 8000  # Reduced sampling rate for faster processing
WINDOW_SIZE = 25  # in ms
WINDOW_STEP = 10  # in ms
N_MELS = 40  # Number of Mel filterbanks
HIDDEN_SIZE = 64  # Reduced hidden size
NUM_LAYERS = 1  # Reduced number of LSTM layers
BATCH_SIZE = 16  # Reduced batch size
EPOCHS = 30  # Fewer epochs for quicker training
LEARNING_RATE = 0.001  # Learning rate for optimizer
SEQ_LENGTH = 100  # Number of frames per sample

# Hardcoded directories
TRAIN_DIR = 'train'
VAL_DIR = 'val'

# Model Definition
class AudioClassifier(nn.Module):
    def __init__(self):
        super(AudioClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size=N_MELS, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, batch_first=True)
        self.fc = nn.Linear(HIDDEN_SIZE, 2)  # Assuming two classes: ads and non-ads
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Get the last time step output
        out = self.fc(out)
        return self.softmax(out)

def train(model, train_loader, criterion, optimizer):
    model.train()
    total_loss = 0
    for data, labels in train_loader:
        data = data.to(DEVICE)
        labels = labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f'Training Loss: {avg_loss:.4f}')

def evaluate(model, val_loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for data, labels in val_loader:
            data = data.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(data)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
    avg_loss = total_loss / len(val_loader)
    accuracy = correct / len(val_loader.dataset)
    print(f'Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')

def infer(model, file_path):
    model.eval()
    features = preprocess_audio(
        file_path, sampling_rate=SAMPLING_RATE, n_mels=N_MELS, seq_length=SEQ_LENGTH)
    features = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(features)
        preds = outputs.argmax(dim=1)
    classes = ['ads', 'non-ads']
    print(f'The audio file "{file_path}" is predicted as: {classes[preds.item()]}')

def main():
    # Initialize model, criterion, optimizer
    model = AudioClassifier().to(DEVICE)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    parser = argparse.ArgumentParser(description='Audio Classification Script')
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'infer'], help='Mode: train or infer')
    parser.add_argument('--file', type=str, help='Path to audio file for inference (required for infer mode)')
    args = parser.parse_args()

    if args.mode == 'train':
        # Prepare datasets
        train_dataset = AudioDataset(TRAIN_DIR, seq_length=SEQ_LENGTH, sampling_rate=SAMPLING_RATE, n_mels=N_MELS)
        val_dataset = AudioDataset(VAL_DIR, seq_length=SEQ_LENGTH, sampling_rate=SAMPLING_RATE, n_mels=N_MELS)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # Training and evaluation loop
        for epoch in range(EPOCHS):
            print(f'Epoch {epoch+1}/{EPOCHS}')
            train(model, train_loader, criterion, optimizer)
            evaluate(model, val_loader, criterion)
            print('-' * 20)

        # Save the trained model
        torch.save(model.state_dict(), 'audio_classifier.pth')
        print('Model saved as audio_classifier.pth')

    elif args.mode == 'infer':
        if args.file is None:
            print('Please provide a file path with --file for inference mode.')
            return
        if not os.path.exists(args.file):
            print(f'File "{args.file}" does not exist.')
            return
        # Load the trained model
        model.load_state_dict(torch.load('audio_classifier.pth', map_location=DEVICE))
        infer(model, args.file)

if __name__ == '__main__':
    main()

