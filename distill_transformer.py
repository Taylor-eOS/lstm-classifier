import os
import time
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from main import infer, get_filename, get_matching_file, get_model, DEVICE, ACCURACY_THRESHOLD, MIN_ACCURACY, CLASSES
from utils import preprocess_audio, create_empty_folder

TRANSFORMER_SAMPLING_RATE = 2000
TRANSFORMER_N_MFCC = 4
TRANSFORMER_FRAMES = 32 #frames
TRANSFORMER_HOP_LENGTH = 512
D_MODEL = 32
MAX_EPOCHS = 20
NUM_LAYERS = 2
NHEAD = 4
DIM_FEEDFORWARD = 128
TRANSFORMER_BATCH_SIZE = 64
LEARNING_RATE = 0.0005
DROPOUT = 0.1
ALPHA = 0.5
TEMP = 2.0
TRAIN_DIR = 'train'
VAL_DIR = 'val'
TEACHER_LOGITS_PATH = 'teacher_logits.pth'

class AudioTransformer(nn.Module):
    def __init__(self):
        super(AudioTransformer, self).__init__()
        self.embedding = nn.Linear(TRANSFORMER_N_MFCC * 3 + 1, D_MODEL)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, 
            nhead=NHEAD, 
            dim_feedforward=DIM_FEEDFORWARD, 
            dropout=DROPOUT,
            batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=NUM_LAYERS)
        self.fc_out = nn.Linear(D_MODEL, 2)
    def forward(self, x):
        #print(f"Input to transformer shape: {x.shape}")  # Should be [64, 32, 13]
        x = self.embedding(x)
        #print(f"Shape after embedding: {x.shape}")  # Should be [64, 32, 32]
        x = self.transformer_encoder(x)
        #print(f"Shape after transformer encoder: {x.shape}")  # Should be [64, 32, 32] if everything is working correctly
        x = x.mean(dim=1)
        #print(f"Shape after mean operation: {x.shape}")  # Should be [64, 32] after averaging over sequence length
        logits = self.fc_out(x)
        return logits

class DistillationDataset(Dataset):
    def __init__(self, data_dir, teacher_logits_path=None):
        self.data = []
        self.labels = []
        self.teacher_logits = None
        if teacher_logits_path is not None:
            if not os.path.exists(teacher_logits_path):
                raise FileNotFoundError(f"Teacher logits file not found at {teacher_logits_path}")
            self.teacher_logits = torch.load(teacher_logits_path)
        for label, class_name in enumerate(CLASSES):
            class_dir = os.path.join(data_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for file_name in os.listdir(class_dir):
                if file_name.lower().endswith('.wav'):
                    file_path = os.path.join(class_dir, file_name)
                    self.data.append(file_path)
                    self.labels.append(label)
        if self.teacher_logits is not None and len(self.teacher_logits) != len(self.data):
            raise ValueError("Number of teacher logits does not match number of data samples.")
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        file_path = self.data[idx]
        label = self.labels[idx]
        if self.teacher_logits is not None:
            teacher_logit = self.teacher_logits[idx]
            return file_path, label, teacher_logit
        else:
            return file_path, label

def preprocess_audio_teacher(file_path):
    feature = preprocess_audio(file_path, sampling_rate=4000, n_mfcc=10, seq_length=128, hop_length=256)
    frame_counts = np.arange(feature.shape[0]).reshape(-1, 1)
    feature = np.concatenate((feature, frame_counts), axis=1)
    return torch.tensor(feature, dtype=torch.float32)

def preprocess_batch(file_paths):
    features_batch = [preprocess_audio_transformer(file) for file in file_paths]
    stacked_batch = torch.stack(features_batch).to(DEVICE)
    #print(f"Stacked batch shape: {stacked_batch.shape}")
    return stacked_batch

def infer_transformer(file_paths, transformer_model):
    transformer_model.eval()
    features_batch = preprocess_batch(file_paths)
    with torch.no_grad():
        logits = transformer_model(features_batch)
    probabilities = torch.softmax(logits, dim=1)
    preds = torch.argmax(probabilities, dim=1).tolist()
    prob_Bs = probabilities[:, 1].tolist()
    pred_classes = [CLASSES[pred] for pred in preds]
    return pred_classes, prob_Bs, logits.cpu().numpy()

def evaluate_transformer(model, val_loader):
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for batch in val_loader:
            if len(batch) == 3:
                _, labels, _ = batch
            else:
                _, labels = batch
            labels = labels.to(DEVICE)
            preds, _, _ = infer(model, batch[0])
            preds = torch.tensor([CLASSES.index(pred) for pred in preds], dtype=torch.long).to(DEVICE)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
    accuracy = total_correct / total_samples
    return accuracy

def get_filename_transformer(epoch):
    epoch = str(epoch)
    return f'transformer_{TRANSFORMER_SAMPLING_RATE}_{TRANSFORMER_N_MFCC}_{TRANSFORMER_FRAMES}_{D_MODEL}_{epoch}.pth'

def precompute_teacher_logits(teacher_model, train_loader, save_path):
    teacher_model.eval()
    all_teacher_logits = []
    with torch.no_grad():
        for batch in train_loader:
            if len(batch) == 2:
                file_paths, _ = batch
            elif len(batch) == 3:
                file_paths, _, _ = batch
            else:
                raise ValueError(f"Unexpected batch size: {len(batch)}")
            features_batch = [preprocess_audio_teacher(file) for file in file_paths]
            features_batch = torch.stack(features_batch).to(DEVICE)
            logits = teacher_model(features_batch)
            all_teacher_logits.append(logits.cpu())
    all_teacher_logits = torch.cat(all_teacher_logits, dim=0)
    torch.save(all_teacher_logits, save_path)
    print(f'Teacher logits saved to {save_path}')

def main_transformer(mode, input_file=None):
    if mode == 'train':
        max_epochs = MAX_EPOCHS
        transformer_model = AudioTransformer().to(DEVICE)
        teacher_filename = get_matching_file(get_filename)
        teacher_model = get_model(teacher_filename).to(DEVICE)
        teacher_model.eval()
        criterion_ce = nn.CrossEntropyLoss()
        criterion_kd = nn.KLDivLoss(reduction='batchmean')
        optimizer = optim.Adam(transformer_model.parameters(), lr=LEARNING_RATE)
        train_dataset_initial = DistillationDataset(TRAIN_DIR)
        train_loader_initial = DataLoader(train_dataset_initial, batch_size=TRANSFORMER_BATCH_SIZE, shuffle=False)
        precompute_teacher_logits(teacher_model, train_loader_initial, TEACHER_LOGITS_PATH)
        train_dataset = DistillationDataset(TRAIN_DIR, teacher_logits_path=TEACHER_LOGITS_PATH)
        val_dataset = DistillationDataset(VAL_DIR)
        train_loader = DataLoader(train_dataset, batch_size=TRANSFORMER_BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=TRANSFORMER_BATCH_SIZE, shuffle=False)
        highest_accuracy = MIN_ACCURACY
        best_model = None
        for epoch in range(max_epochs):
            print(f'Epoch {epoch+1}/{max_epochs}')
            start_time = time.time()
            transformer_model.train()
            batch_losses = []
            for batch in train_loader:
                file_paths, labels, teacher_logits = batch
                labels = labels.to(DEVICE)
                teacher_logits = teacher_logits.to(DEVICE)
                optimizer.zero_grad()
                features_batch = preprocess_batch(file_paths)
                student_logits = transformer_model(features_batch)
                loss_ce = criterion_ce(student_logits, labels)
                loss_kd = criterion_kd(
                    torch.log_softmax(student_logits / TEMP, dim=1),
                    torch.softmax(teacher_logits / TEMP, dim=1)
                ) * (TEMP ** 2)
                loss = ALPHA * loss_ce + (1 - ALPHA) * loss_kd
                loss.backward()
                optimizer.step()
                batch_losses.append(loss.item())
            train_loss = np.mean(batch_losses)
            elapsed_time = time.time() - start_time
            print(f'Elapsed time: {elapsed_time:.0f} seconds')
            accuracy = evaluate_transformer(transformer_model, val_loader)
            filename_transformer = get_filename_transformer(epoch+1)
            if accuracy > MIN_ACCURACY and accuracy > highest_accuracy:
                torch.save(transformer_model.state_dict(), filename_transformer)
                print('Model saved as', os.path.basename(filename_transformer))
                highest_accuracy = accuracy
                if best_model and os.path.exists(best_model):
                    os.remove(best_model)
                best_model = filename_transformer
            else:
                print('Not saving')
            print(f'Train Loss: {train_loss:.4f}')
            print(f'Val Accuracy: {accuracy * 100:.1f}%')
            if accuracy > ACCURACY_THRESHOLD:
                print(f"An accuracy of {accuracy} is above {ACCURACY_THRESHOLD}. Stopping training early.")
                break
            print('')
    elif mode == 'infer':
        if input_file is None:
            raise ValueError("File path is required for inference mode.")
        input_file = [input_file] if isinstance(input_file, str) else input_file
        if not all(os.path.exists(file) for file in input_file):
            missing_files = [file for file in input_file if not os.path.exists(file)]
            for file in missing_files:
                print(f'File {file} does not exist.')
            return
        filename_transformer = get_matching_file(get_filename_transformer)
        transformer_model = AudioTransformer().to(DEVICE)
        transformer_model.load_state_dict(torch.load(filename_transformer, map_location=DEVICE))
        print(f'Loaded model from {filename_transformer}')
        pred_classes, prob_Bs, logits = infer_transformer(input_file, transformer_model)
        if len(input_file) == 1:
            if __name__ == "__main__":
                print(f'Transformer prediction: {pred_classes[0]} with probability {prob_Bs[0]}')
            return pred_classes[0], prob_Bs[0], logits[0]
        else:
            return pred_classes, prob_Bs, logits

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Audio Classification Script')
    parser.add_argument('--f', type=str, help='Path to audio file for inference')
    args = parser.parse_args()
    mode = 'infer' if args.f else 'train'
    main_transformer(mode, input_file=args.f)

