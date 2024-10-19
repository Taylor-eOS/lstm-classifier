import os
import random
from pydub import AudioSegment

def get_audio_files(folder):
    supported_formats = ('.wav')
    return [f for f in os.listdir(folder) if f.lower().endswith(supported_formats)]

def split_audio(audio):
    mid_point = len(audio) // 2
    first_half = audio[:mid_point]
    second_half = audio[mid_point:]
    return first_half, second_half

def create_new_files(file1, file2, folder, output_folder):
    audio1 = AudioSegment.from_file(os.path.join(folder, file1))
    audio2 = AudioSegment.from_file(os.path.join(folder, file2))
    fh1, sh1 = split_audio(audio1)
    fh2, sh2 = split_audio(audio2)
    new_audio1 = sh1 + fh2
    new_audio2 = sh2 + fh1
    base1, ext1 = os.path.splitext(file1)
    base2, ext2 = os.path.splitext(file2)
    new_file1 = f"swap_{base1}_{base2}{ext1}"
    new_file2 = f"swap_{base2}_{base1}{ext2}"
    new_audio1.export(os.path.join(output_folder, new_file1), format=ext1.replace('.', ''))
    new_audio2.export(os.path.join(output_folder, new_file2), format=ext2.replace('.', ''))
    #print(f"Created: {new_file1} and {new_file2}")

def main():
    folder = input("Enter the relative folder name (e.g., 'train/A'): ").strip()
    if not os.path.isdir(folder):
        print(f"Folder '{folder}' does not exist.")
        return
    audio_files = get_audio_files(folder)
    if len(audio_files) < 2:
        print("Not enough audio files to create pairs.")
        return
    random.shuffle(audio_files)
    #output_folder = os.path.join(folder, "augmented")
    output_folder = folder
    os.makedirs(output_folder, exist_ok=True)
    paired = set()
    for i in range(0, len(audio_files) - 1, 2):
        file1 = audio_files[i]
        file2 = audio_files[i + 1]
        pair = tuple(sorted([file1, file2]))
        if pair not in paired:
            create_new_files(file1, file2, folder, output_folder)
            paired.add(pair)
    # Handle odd number of files by pairing the last file with a random file
    if len(audio_files) % 2 != 0:
        last_file = audio_files[-1]
        other_file = random.choice(audio_files[:-1])
        pair = tuple(sorted([last_file, other_file]))
        if pair not in paired:
            create_new_files(last_file, other_file, folder, output_folder)
            paired.add(pair)
    print(f"A lot of new files have been created in {output_folder}")

if __name__ == "__main__":
    main()

