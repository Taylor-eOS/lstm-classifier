import os
import time
import shutil
import argparse
import random
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import AudioDataset, preprocess_audio, create_empty_folder

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAMPLING_RATE = 4000
N_MFCC = 10
SEQ_LENGTH = 128 #frames
HIDDEN_SIZE = 128
HOP_LENGTH = 256
NUM_LAYERS = 2
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
ACCURACY_THRESHOLD = 0.99
MIN_ACCURACY = 0.75
TRAIN_DIR = 'train'
VAL_DIR = 'val'
DATA_PROPORTION = 4

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

def train(model, train_loader, criterion, optimizer):
    train_loader_list = list(train_loader)
    num_batches = len(train_loader_list) // DATA_PROPORTION
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

def infer(model, file_paths):
    model.eval()
    if isinstance(file_paths, str):
        file_paths = [file_paths]
    features_batch = [
        preprocess_audio(
            file,
            sampling_rate=SAMPLING_RATE,
            n_mfcc=N_MFCC,
            seq_length=SEQ_LENGTH,
            hop_length=HOP_LENGTH
        ) for file in file_paths
    ]
    features_batch = np.array(features_batch, dtype=np.float32)
    features_batch = torch.from_numpy(features_batch).to(DEVICE)
    with torch.no_grad():
        logits = model(features_batch)
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

def get_filename(epoch):
    epoch = str(epoch)
    return f'lstm_{SAMPLING_RATE}_{N_MFCC}_{SEQ_LENGTH}_{HIDDEN_SIZE}_{epoch}.pth'

def get_matching_file(filename_func): #get_filename or get_filename_transformer
    matching_files = glob.glob(filename_func('*'))
    if not matching_files:
        raise ValueError("No matching model file found for the current architecture parameters.")
    matching_files.sort(key=os.path.getmtime, reverse=True)
    return matching_files[0]

def get_model(filename=None):
    model = AudioClassifier().to(DEVICE)
    if filename:
        checkpoint = torch.load(filename, map_location=DEVICE, weights_only=True)
        print(f'Loaded model from {filename}')
        model.load_state_dict(checkpoint)
    return model

def main(mode, batch_size=BATCH_SIZE, input_file=None, model=None):
    if mode == 'train':
        max_epochs = int(input("Max. epochs: "))
        model = AudioClassifier().to(DEVICE)
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        train_dataset = AudioDataset(TRAIN_DIR, seq_length=SEQ_LENGTH, sampling_rate=SAMPLING_RATE, n_mfcc=N_MFCC)
        val_dataset = AudioDataset(VAL_DIR, seq_length=SEQ_LENGTH, sampling_rate=SAMPLING_RATE, n_mfcc=N_MFCC)
        train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
        last_loss = 1.0
        highest_accuracy = MIN_ACCURACY
        best_model = None
        print(f'Each epoch uses 1/{DATA_PROPORTION} of the training data')
        for epoch in range(max_epochs):
            print(f'Epoch {epoch+1}/{max_epochs}')
            start_time = time.time()
            train(model, train_loader, criterion, optimizer)
            elapsed_time = time.time() - start_time
            print(f'Elapsed time: {elapsed_time:.0f} seconds')
            loss, accuracy = evaluate(model, val_loader, criterion)
            print(f"Loss delta: {(loss - last_loss) * -100:.2f}%")
            last_loss = loss
            filename = get_filename(epoch+1)
            if accuracy > MIN_ACCURACY and accuracy > highest_accuracy:
                torch.save(model.state_dict(), filename)
                print('Model saved as', os.path.basename(filename))
                highest_accuracy = accuracy
                if best_model and os.path.exists(best_model):
                    os.remove(best_model)
                best_model = filename
            else:
                print('Model not saving')
            if accuracy > ACCURACY_THRESHOLD:
                print(f"Accuracy {accuracy} is above {ACCURACY_THRESHOLD}. Stopping early.")
                break
            print('')
    elif mode == 'infer':
        if input_file is None:
            raise ValueError("File path is required for inference mode.")
        if isinstance(input_file, str):
            input_file = [input_file]
        if model is None:
            filename = get_matching_file(get_filename)
            model = get_model(filename)
        logits, preds, probabilities = infer(model, input_file)
        classes = ['A', 'B']
        pred_classes = [classes[pred.item()] for pred in preds]
        if probabilities.dim() == 1:
            prob_Bs = probabilities[1].item()
        else:
            prob_Bs = probabilities[:, 1].tolist()
        if isinstance(input_file, list):
            if len(input_file) == 1:
                if __name__ == "__main__":
                    print(f'LSTM prediction: {pred_classes[0]}')
                return pred_classes[0], prob_Bs, logits[0]
            else:
                return pred_classes, prob_Bs, logits
        else:
            return pred_classes[0], prob_Bs, logits[0]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Audio Classification Script')
    parser.add_argument('--f', type=str, help='Path to audio file for inference')
    args = parser.parse_args()
    mode = 'infer' if args.f else 'train'
    main(mode, input_file=args.f)

