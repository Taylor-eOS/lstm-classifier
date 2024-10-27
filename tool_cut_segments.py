import os
import re
import sys
from pydub import AudioSegment
from utils import convert_time_to_seconds

def parse_segments(segments_file_path):
    segments_dict = {}
    current_filename = None
    timestamps = []
    with open(segments_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  
            if line.startswith('['):
                if current_filename and timestamps:
                    segments_dict[current_filename] = timestamps
                filename_match = re.match(r'\[(.*?)\]', line)
                if filename_match:
                    current_filename = filename_match.group(1)
                    timestamps = []
                else:
                    print(f"Invalid header format: {line}")
                    continue
            else:
                try:
                    times = list(map(int, line.split()))
                    timestamps.extend(times)
                except ValueError as ve:
                    print(f"Invalid timestamp format in {current_filename}: {line}")
                    continue
        if current_filename and timestamps:
            segments_dict[current_filename] = timestamps
    return segments_dict

def split_audio(input_dir, output_dir, segments_dict):
    for type_dir in ['A', 'B']:
        type_path = os.path.join(output_dir, type_dir)
        os.makedirs(type_path, exist_ok=True)
    for filename, split_points in segments_dict.items():
        input_path = os.path.join(input_dir, f"{filename}.wav")
        if not os.path.isfile(input_path):
            print(f"Input file not found: {input_path}")
            continue
        try:
            audio = AudioSegment.from_wav(input_path)
        except Exception as e:
            print(f"Error loading {input_path}: {e}")
            continue
        split_points_ms = [int(s * 1000) for s in split_points]
        split_points_ms = [0] + split_points_ms + [len(audio)]
        segments = []
        for idx in range(len(split_points_ms) - 1):
            start_ms = split_points_ms[idx]
            end_ms = split_points_ms[idx + 1]
            segment = audio[start_ms:end_ms]
            segments.append(segment)
        base_filename = os.path.splitext(filename)[0]
        for idx, segment in enumerate(segments, start=1):
            type_label = 'A' if idx % 2 != 0 else 'B'
            output_filename = f"{base_filename}_segment_{idx}.wav"
            output_path = os.path.join(output_dir, type_label, output_filename)
            try:
                segment.export(output_path, format="wav")
                print(f"Saved segment: {output_path}")
            except Exception as e:
                print(f"Error saving {output_path}: {e}")

def main():
    input_dir = 'input'
    output_dir = 'raw'
    segments_file = os.path.join(input_dir, 'segments.txt')
    if not os.path.isfile(segments_file):
        print(f"Segments file not found: {segments_file}")
        sys.exit(1)
    segments_dict = parse_segments(segments_file)
    if not segments_dict:
        print("No segments found to process.")
        sys.exit(0)
    split_audio(input_dir, output_dir, segments_dict)
    print("Audio splitting completed.")

if __name__ == "__main__":
    main()
