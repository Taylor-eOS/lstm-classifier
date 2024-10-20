import os
import shutil
import subprocess
from utils import convert_wav, convert_mp3
from process_file import process_file

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    convert_dir = os.path.join(script_dir, 'convert')
    if not os.path.isdir(convert_dir):
        print(f"Error: The directory '{convert_dir}' does not exist.")
        return
    for filename in os.listdir(convert_dir):
        if filename.lower().endswith('.mp3'):
            base_name = os.path.splitext(filename)[0]
            mp3_path = os.path.join(convert_dir, filename)
            wav_path = os.path.join(convert_dir, f"{base_name}.wav")
            processed_wav_path = os.path.join(convert_dir, f"{base_name}_cut.wav")
            output_mp3_path = os.path.join(convert_dir, f"{base_name}_cut.mp3")
            print(f"\nProcessing '{filename}'...")
            try:
                convert_mp3(mp3_path, wav_path)
            except Exception as e:
                print(f"Failed to convert '{filename}' to WAV: {e}")
                continue
            try:
                process_file(wav_path)
                print(f"Successfully processed '{wav_path}'.")
            except Exception as e:
                print(f"Error processing '{wav_path}': {e}")
                continue
            except Exception as e:
                print(f"Unexpected error: {e}")
                continue
            if os.path.isfile(processed_wav_path):
                try:
                    convert_wav(processed_wav_path, output_mp3_path)
                    print(f"Converted processed file to '{output_mp3_path}'.")
                    os.makedirs('export', exist_ok=True)
                    #shutil.move(wav_path, 'export')
                    os.remove(processed_wav_path)
                except Exception as e:
                    print(f"Failed to convert '{processed_wav_path}' to MP3: {e}")
            else:
                print(f"Processed WAV file '{processed_wav_path}' not found.")

if __name__ == "__main__":
    main()

