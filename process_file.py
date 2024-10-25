import os
import argparse
import shutil
import glob
import numpy as np
import torch
from pydub import AudioSegment
from concurrent.futures import ThreadPoolExecutor
from main import infer, BATCH_SIZE, CLASSES, get_model, get_matching_file, get_filename, CHUNK_LENGTH

KEEP_CHUNKS = True
PRINT_PREDICTIONS = True

def forgiving_heuristic(predictions, min_surround_chunks=3, max_flip_length=2):
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
    segments.append((current_label, current_indices))
    for i in range(1, len(segments) - 1):
        label, indices = segments[i]
        prev_label, prev_indices = segments[i - 1]
        next_label, next_indices = segments[i + 1]
        if len(indices) <= max_flip_length:
            if len(prev_indices) >= min_surround_chunks and len(next_indices) >= min_surround_chunks:
                new_label = 'B' if label == 'A' else 'A'
                if indices[0]+1 == indices[-1]+1:
                    print(f"Flipping chunk {indices[0]+1} from {label} to {new_label}")
                else:
                    print(f"Flipping chunks {indices[0]+1}-{indices[-1]+1} from {label} to {new_label}")
                segments[i] = (new_label, indices)
    corrected_predictions = []
    for label, indices in segments:
        for _ in indices:
            corrected_predictions.append(label)
    return corrected_predictions

def second_forgiving_heuristic(predictions):
    corrected_predictions = predictions.copy()
    pattern_length = 9
    for i in range(len(predictions) - pattern_length + 1):
        segment = predictions[i:i + pattern_length]
        if segment[:3] == ['B', 'B', 'B'] and segment[3:6] == ['A', 'B', 'A'] and segment[6:] == ['B', 'B', 'B']:
            print(f"Found ABA pattern at indices {i + 1}-{i + pattern_length}. Correcting to all B.")
            corrected_predictions[i + 3:i + 6] = ['B', 'B', 'B']
        elif segment[:3] == ['A', 'A', 'A'] and segment[3:6] == ['B', 'A', 'B'] and segment[6:] == ['A', 'A', 'A']:
            print(f"Found BAB pattern at indices {i + 1}-{i + pattern_length}. Correcting to all A.")
            corrected_predictions[i + 3:i + 6] = ['A', 'A', 'A']
    return corrected_predictions

def split_audio(file_path, chunk_length_ms=CHUNK_LENGTH, output_folder='chunks'):
    audio = AudioSegment.from_wav(file_path)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    chunks = []
    for i in range(0, len(audio) - chunk_length_ms, chunk_length_ms):
        chunk = audio[i:i + chunk_length_ms]
        chunk_path = os.path.join(output_folder, f"chunk_{(i//chunk_length_ms) + 1}.wav")
        chunk.export(chunk_path, format='wav')
        chunks.append(chunk_path)
    return chunks

def convert_to_wav(input_path):
    if input_path.lower().endswith('.mp3'):
        audio = AudioSegment.from_mp3(input_path)
        wav_path = os.path.splitext(input_path)[0] + '.wav'
        audio.export(wav_path, format='wav')
        return wav_path
    return input_path

def print_timestamps(corrected_predictions, chunk_length_ms, total_length_ms):
    boundary_points_sec = []
    for i in range(1, len(corrected_predictions)):
        if corrected_predictions[i] != corrected_predictions[i - 1]:
            boundary = i * chunk_length_ms / 1000
            boundary_points_sec.append(boundary)
    if boundary_points_sec and boundary_points_sec[0] == 0:
        boundary_points_sec = boundary_points_sec[1:]
    if boundary_points_sec and boundary_points_sec[-1] == total_length_ms / 1000:
        boundary_points_sec = boundary_points_sec[:-1]
    print(' '.join(map(lambda x: str(int(round(x))), boundary_points_sec)))

def main_process(input_file):
    wav_file = convert_to_wav(input_file)
    output_folder = 'chunks_' + os.path.splitext(wav_file)[0]
    chunks = split_audio(wav_file, output_folder=output_folder)
    model = get_model(get_matching_file(get_filename))
    predictions = []
    with ThreadPoolExecutor() as executor:
        chunk_batches = [chunks[i:i + BATCH_SIZE] for i in range(0, len(chunks), BATCH_SIZE)]
        for batch in chunk_batches:
            logits, preds, probabilities = infer(model, batch)
            pred_classes = [CLASSES[pred.item()] for pred in preds]
            predictions.extend(pred_classes)
    #print(f"Combined predictions before heuristic: {predictions}")
    corrected_predictions = forgiving_heuristic(predictions)
    corrected_predictions = second_forgiving_heuristic(corrected_predictions)
    if PRINT_PREDICTIONS:
        print(f"Corrected predictions: {corrected_predictions}")
    final_chunks = [chunks[i] for i, label in enumerate(corrected_predictions) if label == 'B']
    combined = AudioSegment.empty()
    total_length_ms = len(AudioSegment.from_wav(wav_file))
    for chunk in final_chunks:
        combined += AudioSegment.from_wav(chunk)
    output_path = os.path.splitext(input_file)[0] + '_processed.wav'
    combined.export(output_path, format='wav')
    print_timestamps(corrected_predictions, CHUNK_LENGTH, total_length_ms)
    if not KEEP_CHUNKS:
        shutil.rmtree(output_folder)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process and classify audio files.')
    parser.add_argument('--input', type=str, required=True, help='Path to the input audio file.')
    args = parser.parse_args()
    main_process(args.input)

