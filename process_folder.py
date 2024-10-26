import os
import shutil
from utils import convert_wav, convert_mp3
from process_file import main_process

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    convert_dir = os.path.join(script_dir, 'convert')
    os.makedirs(convert_dir, exist_ok=True)
    for filename in os.listdir(convert_dir):
        if filename.lower().endswith('.wav'):
            print(f"This script handles mp3 files. The file \'{filename}\' was not processed.'")
        if filename.lower().endswith('.mp3'):
            base_name = os.path.splitext(filename)[0]
            mp3_path = os.path.join(convert_dir, filename)
            wav_path = os.path.join(convert_dir, f"{base_name}.wav")
            processed_wav_path = os.path.join(convert_dir, f"{base_name}_cut.wav")
            output_mp3_path = os.path.join(convert_dir, f"{base_name}_cut.mp3")
            #print(f"Processing '{filename}'")
            try:
                convert_mp3(mp3_path, wav_path)
            except Exception as e:
                print(f"Failed to convert '{filename}' to WAV: {e}")
                continue
            try:
                main_process(wav_path)
                print(f"Processed '{os.path.basename(wav_path)}'.")
            except Exception as e:
                print(f"Error processing '{wav_path}': {e}")
                continue
            if os.path.isfile(processed_wav_path):
                try:
                    convert_wav(processed_wav_path, output_mp3_path)
                    print(f"Converted processed file to '{os.path.basename(output_mp3_path)}'.")
                    os.remove(processed_wav_path)
                except Exception as e:
                    print(f"Failed to convert '{processed_wav_path}' to MP3: {e}")
            else:
                print(f"Processed WAV file '{processed_wav_path}' not found.")

if __name__ == "__main__":
    main()

