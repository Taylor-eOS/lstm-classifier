import os
import re
import argparse
import shutil
import glob
import numpy as np
import torch
from pydub import AudioSegment
from concurrent.futures import ThreadPoolExecutor
from main import infer, BATCH_SIZE, get_model, get_matching_file, get_filename, CHUNK_LENGTH
from utils import CLASSES

KEEP_CHUNKS = False
PRINT_PREDICTIONS = False
PRINT_UNCORRECTED_PREDICTIONS = False

def first_forgiving_heuristic(predictions):
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
    def flip_outlier(segments, outlier_length):
        min_surround = outlier_length + 1
        for i in range(1, len(segments) - 1):
            label, indices = segments[i]
            prev_label, prev_indices = segments[i - 1]
            next_label, next_indices = segments[i + 1]
            if len(indices) == outlier_length and len(prev_indices) >= min_surround and len(next_indices) >= min_surround:
                new_label = 'B' if label == 'A' else 'A'
                #if indices[0] + 1 == indices[-1] + 1:
                #    print(f"Flipping chunk {indices[0] + 1} from {label} to {new_label}")
                #else:
                #    print(f"Flipping chunks {indices[0] + 1}-{indices[-1] + 1} from {label} to {new_label}")
                segments[i] = (new_label, indices)
        return segments
    for i in range(1, 6): #6 means up to 5
        segments = flip_outlier(segments, i)
    corrected_predictions = []
    for label, indices in segments:
        for _ in indices:
            corrected_predictions.append(label)
    return corrected_predictions

def second_forgiving_heuristic(predictions):
    def apply(predictions, target):
        opposite = 'A' if target == 'B' else 'B'
        for i in range(len(predictions) - 7 + 1):
            segment = predictions[i:i + 7]
            if segment[:2] == [target] * 2 and segment[2:5] == [opposite, target, opposite] and segment[5:] == [target] * 2:
                #print(f"Found {target*2}{opposite}{target}{opposite}{target*2} pattern at indices {i + 1}-{i + 7}. Correcting to all {target}.")
                predictions[i + 2:i + 5] = [target] * 3
        return predictions
    predictions = apply(predictions, 'A')
    predictions = apply(predictions, 'B')
    return predictions

def third_forgiving_heuristic(predictions):
    def apply(predictions, target):
        opposite = 'A' if target == 'B' else 'B'
        pred_str = ''.join(predictions)
        pattern = (
            f"{target}{{3}}"
            f"{opposite}{{1,2}}"
            f"{target}{{1,2}}"
            f"{opposite}{{1,2}}"
            f"{target}{{3}}"
        )
        corrected_str = re.sub(pattern, target * 9, pred_str)
        return list(corrected_str)
    predictions = apply(predictions, 'A')
    predictions = apply(predictions, 'B')
    return predictions

def fourth_forgiving_heuristic(predictions):
    pred_str = ''.join(predictions)
    pattern = r"A{4}BA{1}B{4}"
    corrected_str = re.sub(pattern, "A" * 4 + "B" * 5, pred_str)
    return list(corrected_str)

def run_forgiving_scripts(predictions):
    predictions = fourth_forgiving_heuristic(predictions)
    predictions = third_forgiving_heuristic(predictions)
    predictions = second_forgiving_heuristic(predictions)
    predictions = first_forgiving_heuristic(predictions)
    return predictions

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
    output_folder = os.path.splitext(os.path.basename(input_file))[0]
    output_folder = os.path.join('chunks', output_folder)
    wav_file = convert_to_wav(input_file)
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
    if PRINT_UNCORRECTED_PREDICTIONS:
        print(f"Uncorrected predictions: {predictions}")
    predictions = run_forgiving_scripts(predictions)
    if PRINT_PREDICTIONS:
        print(f"Corrected predictions: {predictions}")
    final_chunks = [chunks[i] for i, label in enumerate(predictions) if label == 'B']
    combined = AudioSegment.empty()
    total_length_ms = len(AudioSegment.from_wav(wav_file))
    for chunk in final_chunks:
        combined += AudioSegment.from_wav(chunk)
    output_path = os.path.splitext(input_file)[0] + '_cut.wav'
    combined.export(output_path, format='wav')
    print_timestamps(predictions, CHUNK_LENGTH, total_length_ms)
    if not KEEP_CHUNKS:
        shutil.rmtree(output_folder)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process and classify audio files.')
    parser.add_argument('--input', type=str, required=True, help='Path to the input audio file.')
    args = parser.parse_args()
    main_process(args.input)

