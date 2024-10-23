import os
import time
import argparse
import numpy as np
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from utils import preprocess_audio, create_empty_folder
from main import main, get_filename, get_matching_file, get_model, DEVICE, ACCURACY_THRESHOLD, MIN_ACCURACY, BATCH_SIZE as TEACHER_BATCH_SIZE

TRANSFORMER_SAMPLING_RATE = 2000
TRANSFORMER_N_MFCC = 4
TRANSFORMER_FRAMES = 32 #frames
TRANSFORMER_HOP_LENGTH = 512
D_MODEL = 32
NHEAD = 4
NUM_LAYERS = 2
DIM_FEEDFORWARD = 128
BATCH_SIZE = TEACHER_BATCH_SIZE
LEARNING_RATE = 0.001
DROPOUT = 0.1
ALPHA = 0.5
TEMP = 2.0
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
            dropout=DROPOUT,
            batch_first=True
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
    def __init__(self, data_dir, teacher_model):
        self.teacher_model = teacher_model
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
        _, _, teacher_logits = main('infer', TEACHER_BATCH_SIZE, file_path, self.teacher_model)
        #teacher_logits = torch.tensor(teacher_logits, dtype=torch.float32).squeeze(0)
        teacher_logits = teacher_logits.clone().detach().float().squeeze(0)
        return features, label, teacher_logits

def preprocess_audio_transformer(file_path):
    #Use your existing preprocess_audio function with different parameters
    feature = preprocess_audio(file_path, sampling_rate=TRANSFORMER_SAMPLING_RATE, n_mfcc=TRANSFORMER_N_MFCC, seq_length=TRANSFORMER_FRAMES, hop_length=TRANSFORMER_HOP_LENGTH)
    #Add frame counts as an extra dimension
    frame_counts = np.arange(feature.shape[0]).reshape(-1, 1)
    feature = np.concatenate((feature, frame_counts), axis=1)
    return feature

def train_transformer(model, train_loader, criterion_ce, criterion_kd, optimizer):
    model.train()
    total_loss = 0.0
    for inputs, labels, teacher_logits in train_loader:
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)
        teacher_logits = teacher_logits.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss_ce = criterion_ce(outputs, labels)
        outputs_soft = nn.functional.log_softmax(outputs / TEMP, dim=1)
        teacher_soft = nn.functional.softmax(teacher_logits / TEMP, dim=1)
        loss_kd = criterion_kd(outputs_soft, teacher_soft) * (TEMP ** 2)
        loss = ALPHA * loss_ce + (1.0 - ALPHA) * loss_kd
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    return avg_loss

def infer_transformer(model, file_path):
    model.eval()
    feature = preprocess_audio_transformer(file_path)
    feature = torch.tensor(feature, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(feature)
    return logits.cpu()

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

def get_filename_transformer(epoch):
    epoch = str(epoch)
    return f'transformer_{TRANSFORMER_SAMPLING_RATE}_{TRANSFORMER_N_MFCC}_{TRANSFORMER_FRAMES}_{epoch}.pth'

def main_transformer(mode, input_file=None):
    transformer_model = AudioTransformer().to(DEVICE)
    criterion_ce = nn.CrossEntropyLoss()
    criterion_kd = nn.KLDivLoss(reduction='batchmean')
    optimizer = optim.Adam(transformer_model.parameters(), lr=LEARNING_RATE)
    last_loss = 0.0
    highest_accuracy = MIN_ACCURACY
    if mode == 'train':
        max_epochs = int(input("Max. epochs: "))
        teacher_model = get_model(get_matching_file(get_filename))
        #print(f'{teacher_model}')
        train_dataset = DistillationDataset(TRAIN_DIR, teacher_model)
        val_dataset = DistillationDataset(VAL_DIR, teacher_model)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        #create_empty_folder('models')
        highest_accuracy = MIN_ACCURACY
        for epoch in range(max_epochs):
            print(f'Epoch {epoch+1}/{max_epochs}')
            start_time = time.time()
            train_loss = train_transformer(transformer_model, train_loader, criterion_ce, criterion_kd, optimizer)
            elapsed_time = time.time() - start_time
            print(f'Elapsed time: {elapsed_time:.0f} seconds')
            accuracy = evaluate_transformer(transformer_model, val_loader)
            filename_transformer = get_filename_transformer(epoch+1)
            #filename_transformer = os.path.join('models', filename_transformer)
            if accuracy > MIN_ACCURACY and accuracy > highest_accuracy:
                torch.save(transformer_model.state_dict(), filename_transformer)
                if os.path.exists(filename_transformer):
                    print('Model saved as', os.path.basename(filename_transformer))
                highest_accuracy = accuracy
            else:
                print('Not saving')
            print(f'Train Loss: {train_loss:.4f}')
            print(f'Val Accuracy: {accuracy * 100:.1f}%')
            if accuracy > ACCURACY_THRESHOLD:
                print(f"Accuracy {accuracy} is above {ACCURACY_THRESHOLD}. Stopping early.")
                break
            print('')
    elif mode == 'infer':
        if input_file is None:
            raise ValueError("File path is required for inference mode.")
        if not os.path.exists(input_file):
            print(f'File "{input_file}" does not exist.')
            return
        filename_transformer = get_matching_file(get_filename_transformer)
        transformer_model.load_state_dict(torch.load(filename_transformer, map_location=DEVICE, weights_only=True))
        print(f'Loaded model from {filename_transformer}')
        logits = infer_transformer(transformer_model, input_file)
        classes = ['A', 'B']
        predicted_class = torch.argmax(logits, dim=1).item()
        prob_B = nn.functional.softmax(logits, dim=1)[:, 1].item()
        pred_class = classes[predicted_class]
        print(f'Transformer prediction: {pred_class} with {prob_B * 100:.2f}% certainty')
        return pred_class, prob_B, logits.cpu().numpy()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Audio Classification Script')
    parser.add_argument('--f', type=str, help='Path to audio file for inference')
    args = parser.parse_args()
    mode = 'infer' if args.f else 'train'
    main_transformer(mode, args.f)

