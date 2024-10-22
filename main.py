import os
import shutil
import glob
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import AudioDataset, preprocess_audio, create_empty_folder

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAMPLING_RATE = 4000
N_MFCC = 10
SEQ_LENGTH = 160 #frames
HIDDEN_SIZE = 128
NUM_LAYERS = 2
MAX_EPOCHS = 8 #default, you will be prompted
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
ACCURACY_THRESHOLD = 0.995
MIN_ACCURACY = 0.8
TRAIN_DIR = 'train'
VAL_DIR = 'val'
MODEL = None

class AudioClassifier(nn.Module):
    def __init__(self):
        super(AudioClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size=N_MFCC*3, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, batch_first=True)
        self.fc = nn.Linear(HIDDEN_SIZE, 2)
        self.softmax = nn.LogSoftmax(dim=1)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return self.softmax(out)

def train(model, train_loader, criterion, optimizer, data_proportion=4):
    train_loader_list = list(train_loader)
    num_batches = len(train_loader_list) // data_proportion
    print(f'Using 1/{data_proportion} of the training data, {num_batches} batches')
    model.train()
    total_loss = 0
    selected_batches = random.sample(train_loader_list, num_batches)
    for data, labels in selected_batches:
        data = data.to(DEVICE)
        labels = labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / num_batches
    print(f'Training Loss: {avg_loss:.3f}')

def infer(model, file_path):
    model.eval()
    features = preprocess_audio(file_path, sampling_rate=SAMPLING_RATE, n_mfcc=N_MFCC, seq_length=SEQ_LENGTH)
    features = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(features)
        preds = logits.argmax(dim=1)
        probabilities = torch.softmax(logits, dim=1)
    return logits, preds, probabilities

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
    print(f'Accuracy: {accuracy * 100:.1f}%')
    print(f'Validation Loss: {avg_loss:.3f}')
    return avg_loss, accuracy

def get_model(filename=None):
    global MODEL
    if MODEL is None:
        MODEL = AudioClassifier().to(DEVICE)
        if filename:
            checkpoint = torch.load(filename, map_location=DEVICE, weights_only=True)
            print(f'Loaded model from {filename}')
            MODEL.load_state_dict(checkpoint)
    return MODEL

def get_filename(epoch):
    return f'lstm_{SAMPLING_RATE}_{N_MFCC}_{SEQ_LENGTH}_{HIDDEN_SIZE}_{epoch}.pth'

def main(mode, input_file=None, model=None):
    if mode == 'train':
        create_empty_folder('models')
        get_model()
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(MODEL.parameters(), lr=LEARNING_RATE)
        last_loss = None
        MAX_EPOCHS = int(input("Max. epochs: "))
        train_dataset = AudioDataset(TRAIN_DIR, seq_length=SEQ_LENGTH, sampling_rate=SAMPLING_RATE, n_mfcc=N_MFCC)
        val_dataset = AudioDataset(VAL_DIR, seq_length=SEQ_LENGTH, sampling_rate=SAMPLING_RATE, n_mfcc=N_MFCC)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        for epoch in range(MAX_EPOCHS):
            print(f'Epoch {epoch+1}/{MAX_EPOCHS}')
            train(MODEL, train_loader, criterion, optimizer)
            loss, accuracy = evaluate(MODEL, val_loader, criterion)
            if last_loss is None:
                print("Delta: higher is better")
            else:
                delta = loss - last_loss
                print(f"Delta: {delta * -100:.2f}%")
            last_loss = loss
            filename = get_filename(str(epoch+1))
            filename = os.path.join('models', filename)
            if accuracy > MIN_ACCURACY:
                torch.save(MODEL.state_dict(), filename)
                if os.path.exists(filename):
                    print('Model saved as', filename)
            if accuracy > ACCURACY_THRESHOLD:
                print(f"Accuracy {accuracy} is above {ACCURACY_THRESHOLD}.")
                print(f"Stopping training early at epoch {epoch}.")
                break
    elif mode == 'infer':
        if input_file is None:
            raise ValueError("File path is required for inference mode.")
        if not os.path.exists(input_file):
            print(f'File "{input_file}" does not exist.')
            return
        if model is None:
            matching_files = glob.glob(get_filename('*'))
            if not matching_files:
                raise ValueError("No matching model file found for the current architecture parameters.")
            filename = matching_files[0]
            model = get_model(filename)
        logits, preds, probabilities = infer(model, input_file)
        classes = ['A', 'B']
        pred_class = classes[preds.item()]
        prob_B = probabilities[:, 1].item()
        if __name__ == "__main__":
            print(f'LSTM prediction: {pred_class} with {prob_B:.2f}')
        return pred_class, prob_B, logits

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Audio Classification Script')
    parser.add_argument('--f', type=str, help='Path to audio file for inference')
    args = parser.parse_args()
    mode = 'infer' if args.f else 'train'
    main(mode, args.f)

