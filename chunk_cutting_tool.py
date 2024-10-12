import os
from pydub import AudioSegment

def split_audio_files(src_folder, dest_folder, chunk_length_ms=10000, min_chunk_length_ms=1000):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    all_files = [f for f in os.listdir(src_folder) if os.path.isfile(os.path.join(src_folder, f))]
    for file_name in all_files:
        src_file = os.path.join(src_folder, file_name)
        dest_subfolder = os.path.join(dest_folder)
        if not os.path.exists(dest_subfolder):
            os.makedirs(dest_subfolder)
        audio = AudioSegment.from_file(src_file)
        duration_ms = len(audio)
        for i in range(0, duration_ms, chunk_length_ms):
            chunk = audio[i:i + chunk_length_ms]
            if len(chunk) < min_chunk_length_ms:
                continue  # Skip chunks shorter than the minimum length
            chunk_name = f"{file_name.split('.')[0]}_chunk_{i // chunk_length_ms}.wav"
            chunk.export(os.path.join(dest_subfolder, chunk_name), format="wav")

src_folders = ['raw/A', 'raw/B']
dest_base = 'raw_cut'
for folder in src_folders:
    src_folder = os.path.join(folder)
    dest_folder = os.path.join(dest_base, os.path.basename(folder))
    split_audio_files(src_folder, dest_folder)

