import os
import shutil
import random
from pydub import AudioSegment

def split_audio_files(src_folder, dest_folder, chunk_length_ms=10240):
    stride_factor = 2
    hop_length_ms = chunk_length_ms // stride_factor
    min_chunk_length_ms = chunk_length_ms // 10
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    all_files = [f for f in os.listdir(src_folder) if os.path.isfile(os.path.join(src_folder, f))]
    for file_name in all_files:
        if file_name.lower().endswith('.wav'):
            src_file = os.path.join(src_folder, file_name)
            dest_subfolder = os.path.join(dest_folder)
            if not os.path.exists(dest_subfolder):
                os.makedirs(dest_subfolder)
            audio = AudioSegment.from_file(src_file)
            duration_ms = len(audio)
            print(f"Processing '{file_name}' of {duration_ms}ms")
            chunk_index = 0
            for i in range(0, duration_ms, hop_length_ms):
                chunk = audio[i:i + chunk_length_ms]
                if len(chunk) < chunk_length_ms:
                    continue
                chunk_name = f"{os.path.splitext(file_name)[0]}_chunk_{chunk_index}.wav"
                chunk.export(os.path.join(dest_subfolder, chunk_name), format="wav")
                chunk_index += 1

def move_random_files(src_folder, dest_folder, percentage=0.1):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    all_files = [f for f in os.listdir(src_folder) if os.path.isfile(os.path.join(src_folder, f))]
    #num_files_to_move = int(len(all_files) * percentage)
    num_files_to_move = int(len(all_files) * percentage / (1 + percentage))
    files_to_move = random.sample(all_files, num_files_to_move)
    for file_name in files_to_move:
        src_file = os.path.join(src_folder, file_name)
        dest_file = os.path.join(dest_folder, file_name)
        shutil.move(src_file, dest_file)

def move_files(src_folder, dst_folder):
    for file_name in os.listdir(src_folder):
        shutil.move(src_file, os.path.join(dst_folder, file_name))

for folder in ['raw/A', 'raw/B']:
    dest_folder = os.path.join('raw/cut', os.path.basename(folder))
    split_audio_files(folder, dest_folder)
move_random_files('raw/cut/A', 'val/A')
move_random_files('raw/cut/B', 'val/B')
shutil.move('raw/cut/A', 'train/A')
shutil.move('raw/cut/B', 'train/B')
shutil.rmtree('raw')

