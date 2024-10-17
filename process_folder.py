import os
import subprocess
from utils import convert_wav, convert_mp3
from process_file import process_file

def main():
    #Define the path to the 'convert' subfolder relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    convert_dir = os.path.join(script_dir, 'convert')
    #Check if the 'convert' directory exists
    if not os.path.isdir(convert_dir):
        print(f"Error: The directory '{convert_dir}' does not exist.")
        return
    #Iterate over all .mp3 files in the 'convert' directory
    for filename in os.listdir(convert_dir):
        if filename.lower().endswith('.mp3'):
            base_name = os.path.splitext(filename)[0]
            mp3_path = os.path.join(convert_dir, filename)
            wav_path = os.path.join(convert_dir, f"{base_name}.wav")
            processed_wav_path = os.path.join(convert_dir, f"{base_name}_cut.wav")
            output_mp3_path = os.path.join(convert_dir, f"{base_name}_cut.mp3")
            print(f"\nProcessing '{filename}'...")
            #Convert MP3 to WAV
            try:
                convert_mp3(mp3_path, wav_path)
            except Exception as e:
                print(f"Failed to convert '{filename}' to WAV: {e}")
                continue
            #Run process_file.py on the WAV file
            try:
                process_file(wav_path)
                print(f"Successfully processed '{wav_path}'.")
            except subprocess.CalledProcessError as e:
                print(f"Error processing '{wav_path}': {e}")
                continue
            except Exception as e:
                print(f"Unexpected error: {e}")
                continue
            #Convert the processed WAV back to MP3 with '_cut' suffix
            if os.path.isfile(processed_wav_path):
                try:
                    convert_wav(processed_wav_path, output_mp3_path)
                    print(f"Converted processed file to '{output_mp3_path}'.")
                    os.remove(wav_path)
                    os.remove(processed_wav_path)
                except Exception as e:
                    print(f"Failed to convert '{processed_wav_path}' to MP3: {e}")
            else:
                print(f"Processed WAV file '{processed_wav_path}' not found.")

if __name__ == "__main__":
    main()

