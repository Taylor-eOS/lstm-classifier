import os
from pydub import AudioSegment

def split_audio_files(src_folder, dest_folder, chunk_length_ms=10000):
    # Ensure destination folder exists
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    # Get a list of all files in the source directory
    all_files = [f for f in os.listdir(src_folder) if os.path.isfile(os.path.join(src_folder, f))]

    # Process each audio file
    for file_name in all_files:
        src_file = os.path.join(src_folder, file_name)
        dest_subfolder = os.path.join(dest_folder)
        if not os.path.exists(dest_subfolder):
            os.makedirs(dest_subfolder)
        
        # Load the audio file
        audio = AudioSegment.from_file(src_file)
        duration_ms = len(audio)

        # Split into chunks
        for i in range(0, duration_ms, chunk_length_ms):
            chunk = audio[i:i + chunk_length_ms]
            chunk_name = f"{file_name.split('.')[0]}_chunk_{i // chunk_length_ms}.wav"
            chunk.export(os.path.join(dest_subfolder, chunk_name), format="wav")

# Paths to the source and destination directories
src_folders = ['raw/A', 'raw/B']
dest_base = 'raw_cut'

# Process each folder (A and B)
for folder in src_folders:
    src_folder = os.path.join(folder)
    dest_folder = os.path.join(dest_base, os.path.basename(folder))
    split_audio_files(src_folder, dest_folder)

