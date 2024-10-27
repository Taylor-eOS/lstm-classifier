import os
import shutil
import random
from pydub import AudioSegment
from process_file import CHUNK_LENGTH, CLASSES

VALIDATION_SET_RATIO = 0.07
DEBUG = False

def split_and_collect(src_folder, chunk_length_ms):
    if DEBUG: print(f"Starting split_and_collect for source: {os.path.abspath(src_folder)}")
    hop_length_ms = chunk_length_ms // 2
    all_chunks = []
    all_files = [f for f in os.listdir(src_folder) if os.path.isfile(os.path.join(src_folder, f))]
    print(f"Found {len(all_files)} files in {os.path.abspath(src_folder)}")
    for file_name in all_files:
        if file_name.lower().endswith('.wav'):
            src_file = os.path.abspath(os.path.join(src_folder, file_name))
            if DEBUG: print(f"Loading file: {src_file}")
            try:
                audio = AudioSegment.from_file(src_file)
                if DEBUG: print(f"Successfully loaded {file_name}")
            except Exception as e:
                print(f"Error loading {src_file}: {e}")
                continue
            duration_ms = len(audio)
            print(f"Processing {file_name}")
            for chunk_index, i in enumerate(range(0, duration_ms, hop_length_ms)):
                chunk = audio[i:i + chunk_length_ms]
                if len(chunk) < chunk_length_ms:
                    if DEBUG: print(f"Skipping short end chunk {chunk_index} of {file_name}")
                    continue
                chunk_name = f"{os.path.splitext(file_name)[0]}_chunk_{chunk_index}.wav"
                chunk_data = (chunk_name, chunk)
                all_chunks.append(chunk_data)
    if DEBUG: print(f"Collected {len(all_chunks)} chunks from {os.path.abspath(src_folder)}")
    return all_chunks

def distribute_chunks(all_chunks, train_folder, val_folder, val_percentage):
    if DEBUG: print(f"Starting distribution of {len(all_chunks)} chunks")
    random.shuffle(all_chunks)
    num_val = int(len(all_chunks) * val_percentage)
    val_chunks = all_chunks[:num_val]
    train_chunks = all_chunks[num_val:]
    print(f"Assigning {len(train_chunks)} chunks to training folder and {len(val_chunks)} chunks to validation folder")
    if not os.path.exists(train_folder):
        os.makedirs(train_folder)
        if DEBUG: print(f"Created training folder: {os.path.abspath(train_folder)}")
    else:
        if DEBUG: print(f"Training folder already exists: {os.path.abspath(train_folder)}")
    if not os.path.exists(val_folder):
        os.makedirs(val_folder)
        if DEBUG: print(f"Created validation folder: {os.path.abspath(val_folder)}")
    else:
        if DEBUG: print(f"Validation folder already exists: {os.path.abspath(val_folder)}")
    for chunk_name, chunk in train_chunks:
        destination_path = os.path.abspath(os.path.join(train_folder, chunk_name))
        if DEBUG: print(f"Exporting {chunk_name} to training folder at {destination_path}")
        try:
            chunk.export(destination_path, format="wav")
            if os.path.exists(destination_path):
                if DEBUG: print(f"Successfully exported {chunk_name} to training folder")
            else:
                print(f"Failed to export {chunk_name} to training folder")
        except Exception as e:
            print(f"Error exporting chunk {chunk_name}: {e}")
    for chunk_name, chunk in val_chunks:
        destination_path = os.path.abspath(os.path.join(val_folder, chunk_name))
        if DEBUG: print(f"Exporting {chunk_name} to validation folder at {destination_path}")
        try:
            chunk.export(destination_path, format="wav")
            if os.path.exists(destination_path):
                if DEBUG: print(f"Successfully exported {chunk_name} to validation folder")
            else:
                print(f"Failed to export {chunk_name} to validation folder")
        except Exception as e:
            print(f"Error exporting chunk {chunk_name}: {e}")
    if DEBUG: print("Distribution of chunks completed")

if not os.path.exists('raw'):
    print("The segment folder does not exist. Run the 'tool_cut_segments' script first.")
train_val_percentage = VALIDATION_SET_RATIO
print(f"Validation set ratio: {VALIDATION_SET_RATIO * 100}%")
if DEBUG: print("Starting the distribution process")
for category in CLASSES:
    src = os.path.join('raw', category)
    train_dest = os.path.join('train', category)
    val_dest = os.path.join('val', category)
    if DEBUG: print(f"Processing category {category}")
    all_chunks = split_and_collect(src, CHUNK_LENGTH)
    distribute_chunks(all_chunks, train_dest, val_dest, train_val_percentage)
print("Distribution process completed")
shutil.rmtree('raw')

