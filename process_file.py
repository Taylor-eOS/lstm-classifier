import os
import time
import math
import argparse
import tempfile
from pydub import AudioSegment
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils import convert_mp3, convert_wav
from main import main

def run_inference(file_path):
    try:
        pred_class, prob_B, logits = main(mode='infer', file=file_path)
        if pred_class == 'A':
            return 'A'
        elif pred_class == 'B':
            return 'B'
        else:
            print(f'Error: Unexpected inference output for {file_path}')
            return 'B'
    except Exception as e:
        print(f'Error running inference on {file_path}: {e}')
        return 'B'

#Processes a single audio chunk: saves it temporarily, runs inference, and deletes the temp file.
def process_chunk(chunk_index, chunk):
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
        chunk.export(temp_file.name, format='wav')
        temp_file_path = temp_file.name
    predicted_class = run_inference(temp_file_path)
    os.remove(temp_file_path)
    return (chunk_index, predicted_class)

#Applies a heuristic to correct misclassifications and enforce the expected segment pattern.
def apply_forgiving_heuristic(predictions, min_surround_chunks=3, max_flip_length=2):
    # Step 1: Group consecutive identical labels into segments
    segments = []
    current_label = predictions[0]
    current_indices = [0]
    for i in range(1, len(predictions)):
        if predictions[i] == current_label:
            current_indices.append(i)
        else:
            segments.append((current_label, current_indices))
            current_label = predictions[i]
            current_indices = [i]
    segments.append((current_label, current_indices))  # Add the last segment
    # Step 2: Iterate through segments to find and flip misclassifications
    for i in range(1, len(segments) - 1):
        label, indices = segments[i]
        prev_label, prev_indices = segments[i - 1]
        next_label, next_indices = segments[i + 1]
        # Check if the current segment is short and surrounded by opposite types
        if len(indices) <= max_flip_length and prev_label != label and next_label != label:
            # Check if surrounding segments have enough chunks
            if len(prev_indices) >= min_surround_chunks and len(next_indices) >= min_surround_chunks:
                # Flip the label
                new_label = 'B' if label == 'A' else 'A'
                print(f"Flipping from {label} to {new_label}: Chunks {indices[0]+1}-{indices[-1]+1}")
                segments[i] = (new_label, indices)
    # Step 3: Reconstruct the corrected predictions list
    corrected_predictions = []
    for label, indices in segments:
        for _ in indices:
            corrected_predictions.append(label)
    return corrected_predictions

def reconstruct_audio(predictions, chunks, input_file=None, desired_label='B'):
    total_time = 0
    A_segments = []
    in_A_segment = False
    segment_start = 0
    combined_audio = None
    episode_number = time.strftime("%d-%H%M")
    base_filename = os.path.splitext(os.path.basename(input_file))[0] if input_file else 'unknown'
    txt_file_path = f'timestamps_{base_filename}.txt'
    for i, chunk in enumerate(chunks):
        label = predictions[i]
        duration = len(chunk)
        if label == 'A':
            if not in_A_segment:
                in_A_segment = True
                segment_start = total_time
        else:
            if in_A_segment:
                in_A_segment = False
                A_segments.append((segment_start, total_time))
        if label == desired_label:
            combined_audio = chunk if combined_audio is None else combined_audio + chunk
        total_time += duration
    if in_A_segment:
        A_segments.append((segment_start, total_time))
    for start_ms, end_ms in A_segments:
        start_min, start_sec = divmod(start_ms // 1000, 60)
        end_min, end_sec = divmod(end_ms // 1000, 60)
    return combined_audio

def print_timestamps(corrected_predictions, chunk_length_sec, total_length_sec):
    boundary_points_sec = []
    for i in range(1, len(corrected_predictions)):
        if corrected_predictions[i] != corrected_predictions[i - 1]:
            boundary = i * chunk_length_sec
            boundary_points_sec.append(boundary)
    if boundary_points_sec and boundary_points_sec[0] == 0:
        boundary_points_sec = boundary_points_sec[1:]
    if boundary_points_sec and boundary_points_sec[-1] == total_length_sec:
        boundary_points_sec = boundary_points_sec[:-1]
    print(' '.join(map(str, boundary_points_sec)))

#Processes the input audio file by removing segments classified as type A and combining the remaining type B segments into a new audio file.
def process_audio(input_file, chunk_length_sec=10):
    try:
        audio = AudioSegment.from_wav(input_file)
    except Exception as e:
        print(f'Error loading audio file {input_file}: {e}')
        return
    total_length_ms = len(audio)
    total_length_sec = total_length_ms / 1000
    chunk_length_ms = chunk_length_sec * 1000
    num_chunks = total_length_ms // chunk_length_ms
    num_chunks = int(num_chunks)
    filename, file_extension = os.path.splitext(input_file)
    output_file = f'{filename}_cut{file_extension}'
    print(f'Processing audio file: {input_file}')
    print(f'Total length: {total_length_sec:.2f} seconds')
    print(f'Chunk length: {chunk_length_sec} seconds')
    print(f'Number of full chunks: {num_chunks}')
    chunks = []
    for i in range(num_chunks):
        start_ms = i * chunk_length_ms
        end_ms = start_ms + chunk_length_ms
        chunk = audio[start_ms:end_ms]
        chunks.append(chunk)
    predictions = [None] * num_chunks
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_chunk, i, chunk): i for i, chunk in enumerate(chunks)}
        for future in as_completed(futures):
            chunk_index, predicted_class = future.result()
            predictions[chunk_index] = predicted_class
            print(f'{chunk_index + 1:03}/{num_chunks}: {predicted_class}')
    corrected_predictions = apply_forgiving_heuristic(predictions)
    combined_audio = reconstruct_audio(corrected_predictions, chunks, input_file)
    if combined_audio:
        try:
            combined_audio.export(output_file, format='wav')
            print(f'\nCombined type B audio saved')
        except Exception as e:
            print(f'Error saving combined audio: {e}')
    else:
        print('\nNo type B segments found after applying heuristic.')
    print_timestamps(corrected_predictions, chunk_length_sec, total_length_sec)

def process_file(input_path):
    if os.path.isfile(input_path):
        if input_path.endswith('.mp3'):
            wav_path = input_path.replace('.mp3', '.wav')
            utils.convert_mp3(input_path, wav_path)
            input_path = wav_path
        print('Warning: The script is currently only equipped to handle files under 1 hour length.')
        process_audio(input_path)
    else:
        print(f'Input file {input_path} given to process_file does not exist.')
        return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', required=True, help='Input file path')
    args = parser.parse_args()
    process_file(args.file)

