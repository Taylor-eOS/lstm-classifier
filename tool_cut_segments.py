import os
import re
import sys
from pydub import AudioSegment
from utils import convert_time_to_seconds

#This file splits the input files into the segments marked in the segmetns file.
#Set the lines per file variable to match your file format.

def parse_segments(segments_file_path):
    segments_dict = {}
    current_filename = None
    timestamps = []
    with open(segments_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # Skip empty lines
            if line.startswith('['):
                # Save the previous file's timestamps without omitting any
                if current_filename and timestamps:
                    segments_dict[current_filename] = timestamps
                # Extract the new filename
                filename_match = re.match(r'\[(.*?)\]', line)
                if filename_match:
                    current_filename = filename_match.group(1)
                    timestamps = []
                else:
                    print(f"Invalid header format: {line}")
                    continue
            else:
                # Read space-separated times as integers (no conversion to milliseconds)
                try:
                    times = list(map(int, line.split()))
                    timestamps.extend(times)
                except ValueError as ve:
                    print(f"Invalid timestamp format in {current_filename}: {line}")
                    continue
        # Save the last file's timestamps without omitting any
        if current_filename and timestamps:
            segments_dict[current_filename] = timestamps
    return segments_dict
#Splits audio files into segments based on provided split points

def split_audio(input_dir, output_dir, segments_dict):
    # Create 'A' and 'B' directories inside the output directory if they don't exist
    for type_dir in ['A', 'B']:
        type_path = os.path.join(output_dir, type_dir)
        os.makedirs(type_path, exist_ok=True)
    # Iterate over each file and its split points
    for filename, split_points in segments_dict.items():
        input_path = os.path.join(input_dir, f"{filename}.wav")
        # Check if the input file exists
        if not os.path.isfile(input_path):
            print(f"Input file not found: {input_path}")
            continue
        try:
            # Load the audio file
            audio = AudioSegment.from_wav(input_path)
        except Exception as e:
            print(f"Error loading {input_path}: {e}")
            continue
        # Convert split points from seconds to milliseconds
        split_points_ms = [int(s * 1000) for s in split_points]
        # Prepend 0 ms (start of audio) and append len(audio) ms (end of audio)
        split_points_ms = [0] + split_points_ms + [len(audio)]
        segments = []
        # Iterate through split points to create segments
        for idx in range(len(split_points_ms) - 1):
            start_ms = split_points_ms[idx]
            end_ms = split_points_ms[idx + 1]
            segment = audio[start_ms:end_ms]
            segments.append(segment)
        # Extract base filename without extension for naming
        base_filename = os.path.splitext(filename)[0]
        # Export each segment to the appropriate directory
        for idx, segment in enumerate(segments, start=1):
            # Determine segment type ('A' or 'B')
            type_label = 'A' if idx % 2 != 0 else 'B'
            # Construct output filename
            output_filename = f"{base_filename}_segment_{idx}.wav"
            output_path = os.path.join(output_dir, type_label, output_filename)
            try:
                # Export the segment
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

