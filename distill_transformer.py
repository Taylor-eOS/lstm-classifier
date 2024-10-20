import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import glob
from utils import preprocess_audio
from main import main

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TRANSFORMER_SAMPLING_RATE = 2000
TRANSFORMER_N_MFCC = 4
TRANSFORMER_FRAMES = 20 #frames
D_MODEL = 32
NHEAD = 4
NUM_LAYERS = 2
DIM_FEEDFORWARD = 128
DROPOUT = 0.1
BATCH_SIZE = 32
LEARNING_RATE = 0.001
MAX_EPOCHS = 5 #default, you will be prompted
ALPHA = 0.5 #Weight for the student loss
TEMP = 2.0 #Temperature for softening probabilities
TRAIN_DIR = 'train'
VAL_DIR = 'val'

class AudioTransformer(nn.Module):
    def __init__(self):
        super(AudioTransformer, self).__init__()
        self.embedding = nn.Linear(TRANSFORMER_N_MFCC * 3 + 1, D_MODEL)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, 
            nhead=NHEAD, 
            dim_feedforward=DIM_FEEDFORWARD, 
            dropout=DROPOUT
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=NUM_LAYERS)
        self.fc_out = nn.Linear(D_MODEL, 2)

    def forward(self, x):
        x = self.embedding(x)  # (batch_size, seq_length, D_MODEL)
        x = x.permute(1, 0, 2)  # (seq_length, batch_size, D_MODEL)
        x = self.transformer_encoder(x)
        x = x.mean(dim=0)  # (batch_size, D_MODEL)
        logits = self.fc_out(x)  # (batch_size, 2)
        return logits

class DistillationDataset(Dataset):
    def __init__(self, data_dir):
        self.data = []
        self.labels = []
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
        #Transformer input features
        features = preprocess_audio_transformer(file_path)
        features = torch.tensor(features, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        #Get teacher logits by calling the LSTM's main function
        _, _, teacher_logits = main('infer', file_path, True)
        teacher_logits = torch.tensor(teacher_logits, dtype=torch.float32).squeeze(0)
        return features, label, teacher_logits

def preprocess_audio_transformer(file_path):
    #Use your existing preprocess_audio function with different parameters
    feature = preprocess_audio(file_path, sampling_rate=TRANSFORMER_SAMPLING_RATE, n_mfcc=TRANSFORMER_N_MFCC, seq_length=TRANSFORMER_FRAMES)
    #Add frame counts as an extra dimension
    frame_counts = np.arange(feature.shape[0]).reshape(-1, 1)
    feature = np.concatenate((feature, frame_counts), axis=1)
    return feature

def train_transformer(model, train_loader, criterion_ce, criterion_kd, optimizer, alpha, temperature):
    model.train()
    total_loss = 0.0
    for inputs, labels, teacher_logits in train_loader:
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)
        teacher_logits = teacher_logits.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        #Student loss (cross-entropy with true labels)
        loss_ce = criterion_ce(outputs, labels)
        #Distillation loss (KL divergence with teacher logits)
        outputs_soft = nn.functional.log_softmax(outputs / temperature, dim=1)
        teacher_soft = nn.functional.softmax(teacher_logits / temperature, dim=1)
        loss_kd = criterion_kd(outputs_soft, teacher_soft) * (temperature ** 2)
        #Combined loss
        loss = alpha * loss_ce + (1.0 - alpha) * loss_kd
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    return avg_loss

def evaluate_transformer(model, val_loader):
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for inputs, labels, _ in val_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, dim=1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
    accuracy = total_correct / total_samples
    return accuracy

def main_transformer(mode, file=None):
    transformer_model = AudioTransformer().to(DEVICE)
    criterion_ce = nn.CrossEntropyLoss()
    criterion_kd = nn.KLDivLoss(reduction='batchmean')
    optimizer = optim.Adam(transformer_model.parameters(), lr=LEARNING_RATE)
    last_loss = None
    def get_filename(epoch):
        return f'transformer_{TRANSFORMER_SAMPLING_RATE}_{TRANSFORMER_N_MFCC}_{TRANSFORMER_FRAMES}_{epoch}.pth'
    if mode == 'train':
        MAX_EPOCHS = int(input("Max. epochs: "))
        train_dataset = DistillationDataset(TRAIN_DIR)
        val_dataset = DistillationDataset(VAL_DIR)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        for epoch in range(MAX_EPOCHS):
            print(f'Epoch {epoch+1}/{MAX_EPOCHS}')
            train_loss = train_transformer(transformer_model, train_loader, criterion_ce, criterion_kd, optimizer, ALPHA, TEMP)
            val_accuracy = evaluate_transformer(transformer_model, val_loader)
            if last_loss is None:
                print(f"Delta: higher is better")
            else:
                delta = train_loss - last_loss
                print(f"Delta: {delta * -100:.2f}%")
            last_loss = train_loss
            filename = get_filename(str(epoch+1))
            torch.save(transformer_model.state_dict(), filename)
            if os.path.exists(filename):
                print('Model saved as', filename)
            print(f'Train Loss: {train_loss:.4f} | Val Accuracy: {val_accuracy:.4f}')
            print('-' * 3)
    elif mode == 'infer':
        if file is None:
            raise ValueError("File path is required for inference mode.")
        if not os.path.exists(file):
            print(f'File "{file}" does not exist.')
            return
        expected_filename = get_filename('*')
        matching_files = glob.glob(expected_filename)
        if not matching_files:
            raise ValueError("No matching model file found for the current architecture parameters.")
        filename = matching_files[0]
        transformer_model.load_state_dict(torch.load(filename, map_location=DEVICE, weights_only=True))
        print(f'Loaded model from {filename}')
        logits = infer_transformer(transformer_model, file)
        classes = ['A', 'B']
        predicted_class = torch.argmax(logits, dim=1).item()
        prob_B = nn.functional.softmax(logits, dim=1)[:, 1].item()
        pred_class = classes[predicted_class]
        print(f'Transformer prediction: {pred_class} with {prob_B}')
        return pred_class, prob_B, logits.cpu().numpy()

def infer_transformer(model, file_path):
    model.eval()
    feature = preprocess_audio_transformer(file_path)
    feature = torch.tensor(feature, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(feature)
    return logits.cpu()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Audio Classification Script')
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'infer'], help='Mode: train or infer')
    parser.add_argument('--file', type=str, help='Path to audio file for inference (required for infer mode)')
    args = parser.parse_args()
    main_transformer(args.mode, args.file)

