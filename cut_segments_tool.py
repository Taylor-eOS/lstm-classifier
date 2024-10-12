import os
import re
import sys
from pydub import AudioSegment
from utils import convert_time_to_seconds

def parse_segments(segments_file_path):
    segments_dict = {}
    with open(segments_file_path, 'r') as f:
        lines = f.readlines()
        for i in range(0, len(lines), 6):
            header = lines[i].strip()
            filename_match = re.match(r'\[(.*?)\]', header)
            if filename_match:
                filename = filename_match.group(1)
                timestamps = []
                for j in range(1, 6):
                    if i + j >= len(lines):
                        print(f"Unexpected end of file while parsing {filename}.")
                        sys.exit(1)
                    time_range = lines[i + j].strip()
                    if '-' not in time_range:
                        print(f"Invalid time range format in {filename}: '{time_range}'")
                        sys.exit(1)
                    start, end = time_range.split('-')
                    start_sec = convert_time_to_seconds(start)
                    end_sec = convert_time_to_seconds(end)
                    timestamps.extend([start_sec * 1000, end_sec * 1000])
                if len(timestamps) < 2:
                    print(f"Not enough timestamps for {filename}.")
                    continue
                split_points = sorted(timestamps[1:-1])
                segments_dict[filename] = split_points
    return segments_dict

def split_audio(input_dir, output_dir, segments_dict):
    for type_dir in ['A', 'B']:
        type_path = os.path.join(output_dir, type_dir)
        if not os.path.exists(type_path):
            os.makedirs(type_path)
    for filename, split_points in segments_dict.items():
        input_path = os.path.join(input_dir, filename)
        input_path = f'{input_path}.wav'
        if not os.path.isfile(input_path):
            print(f"Input file not found: {input_path}")
            continue
        try:
            audio = AudioSegment.from_wav(input_path)
        except Exception as e:
            print(f"Error loading {input_path}: {e}")
            continue
        split_points = [0] + split_points + [len(audio)]
        segments = []
        for idx in range(len(split_points) - 1):
            start_ms = split_points[idx]
            end_ms = split_points[idx + 1]
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
    output_dir = 'output'
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

