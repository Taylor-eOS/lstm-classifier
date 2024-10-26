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
from torch.utils.data import DataLoader, Subset
from utils import AudioDataset, preprocess_audio, create_empty_folder, CLASSES

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAMPLING_RATE = 4000
N_MFCC = 10
SEQ_LENGTH = 128  #aka. frames
HOP_LENGTH = 256
HIDDEN_SIZE = 128
MAX_EPOCHS = 20
NUM_LAYERS = 2
BATCH_SIZE = 32
LEARNING_RATE = 0.00005
STOPPING = False
ACCURACY_THRESHOLD = 0.99
MIN_ACCURACY = 0.7
TRAIN_DIR = 'train'
VAL_DIR = 'val'
CHUNK_LENGTH = 8192
BATCHES_PER_EPOCH = 22

class AudioClassifier(nn.Module):
    def __init__(self):
        super(AudioClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size=N_MFCC * 3, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, batch_first=True)
        self.fc = nn.Linear(HIDDEN_SIZE, 2)
        self.softmax = nn.LogSoftmax(dim=1)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return self.softmax(out)

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

def infer(model, file_paths):
    model.eval()
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

def train(model, train_dataset, criterion, optimizer, batch_size=BATCH_SIZE, batches_per_epoch=BATCHES_PER_EPOCH):
    total_samples = len(train_dataset)
    if batches_per_epoch is None or batches_per_epoch * batch_size > total_samples:
        subset_indices = list(range(total_samples))
    else:
        num_samples = batches_per_epoch * batch_size
        subset_indices = random.sample(range(total_samples), num_samples)
    subset = Subset(train_dataset, subset_indices)
    subset_loader = DataLoader(subset, batch_size, shuffle=True)
    model.train()
    total_loss = 0.0
    for data, labels in subset_loader:
        data = data.to(DEVICE)
        labels = labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(subset_loader)
    print(f'Training Loss: {avg_loss:.3f}')

def get_filename(epoch):
    epoch = str(epoch)
    return f'lstm_{SAMPLING_RATE}_{N_MFCC}_{SEQ_LENGTH}_{HIDDEN_SIZE}_{epoch}.pth'

def get_matching_file(filename_func):  #get_filename or get_filename_transformer
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
        max_epochs = MAX_EPOCHS
        model = AudioClassifier().to(DEVICE)
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        train_dataset = AudioDataset(TRAIN_DIR, seq_length=SEQ_LENGTH, sampling_rate=SAMPLING_RATE, n_mfcc=N_MFCC)
        val_dataset = AudioDataset(VAL_DIR, seq_length=SEQ_LENGTH, sampling_rate=SAMPLING_RATE, n_mfcc=N_MFCC)
        val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
        print(f'Total samples: {len(train_dataset)}')
        print(f'Batches: {len(train_dataset) / BATCH_SIZE}')
        last_loss = 1.0
        highest_accuracy = MIN_ACCURACY
        best_model = None
        print(f'Batch sampling: {BATCHES_PER_EPOCH} batches per epoch')
        for epoch in range(max_epochs):
            print(f'Epoch {epoch+1}/{max_epochs}')
            start_time = time.time()
            train(model, train_dataset, criterion, optimizer)
            elapsed_time = time.time() - start_time
            print(f'Elapsed time: {elapsed_time:.0f} seconds')
            loss, accuracy = evaluate(model, val_loader, criterion)
            print(f"Loss delta: {(loss - last_loss) * -100:.2f}%")
            last_loss = loss
            filename = get_filename(epoch + 1)
            if accuracy >= MIN_ACCURACY and accuracy >= highest_accuracy:
                torch.save(model.state_dict(), filename)
                print('Model saved as', os.path.basename(filename))
                highest_accuracy = accuracy
                if best_model and os.path.exists(best_model):
                    os.remove(best_model)
                best_model = filename
            else:
                print('Not saving model with lower accuracy')
            if accuracy >= ACCURACY_THRESHOLD and STOPPING:
                print(f"Accuracy {accuracy} is above {ACCURACY_THRESHOLD}. Stopping early.")
                break
            print('')

    elif mode == 'infer':
        if input_file is None:
            raise ValueError("File path is required for inference mode.")
        input_file = [input_file] if isinstance(input_file, str) else input_file
        model = get_model(get_matching_file(get_filename))
        logits, preds, probabilities = infer(model, input_file)
        pred_classes = [CLASSES[pred.item()] for pred in preds]
        if probabilities.dim() == 1:
            probabilities = probabilities.unsqueeze(0)
        prob_Bs = probabilities[:, 1].tolist()
        for i, (pred_class, prob_B) in enumerate(zip(pred_classes, prob_Bs), 1):
            print(f'LSTM prediction: {pred_class} with B probability {prob_B:.2f}')
        if len(input_file) == 1:
            return pred_classes[0], prob_Bs[0], logits[0]
        else:
            return pred_classes, prob_Bs, logits

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Audio Classification Script')
    parser.add_argument('f', type=str, nargs='?', help='Path to audio file for inference')
    args = parser.parse_args()
    mode = 'infer' if args.f else 'train'
    main(mode, input_file=args.f)

