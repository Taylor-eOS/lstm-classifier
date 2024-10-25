import os
import re
import time
import argparse
from concurrent.futures import ThreadPoolExecutor
from main import main, BATCH_SIZE
from distill_transformer import main_transformer

def run_inference(file_path):
    pred_class, prob_B, logits = main(mode='infer', input_file=file_path)
    return pred_class, prob_B, logits

def run_inference_transformer(file_path):
    pred_class, prob_B, logits = main_transformer(mode='infer', input_file=file_path)
    return pred_class, prob_B, logits

def evaluate_accuracy(directory, use_transformer, batch_size=BATCH_SIZE):
    files = [file_name for file_name in os.listdir(directory) if file_name.endswith('.wav')]
    if not files:
        print("No .wav files found in the directory.")
        return
    total_files = len(files)
    correct_predictions = 0
    for i in range(0, total_files, batch_size):
        batch_files = files[i:i + batch_size]
        file_paths = [os.path.join(directory, file_name) for file_name in batch_files]
        true_classes = [file_name.split('_')[-1][0].upper() for file_name in batch_files]
        if use_transformer:
            pred_classes, _, _ = run_inference_transformer(file_paths)
        else:
            pred_classes, _, _ = run_inference(file_paths)
        pred_classes = [pred.upper() for pred in pred_classes]
        for file_name, pred_class, true_class in zip(batch_files, pred_classes, true_classes):
            file_number_match = re.search(r'\d+', file_name)
            file_number = file_number_match.group() if file_number_match else "N/A"
            if pred_class == true_class:
                print(f'{file_number:2} {pred_class} correct')
                correct_predictions += 1
            else:
                print(f'{file_number} {pred_class} wrong')
    accuracy = (correct_predictions / total_files) * 100
    print(f'Accuracy: {accuracy:.2f}% ({correct_predictions}/{total_files} correct predictions)')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch inference script")
    parser.add_argument('--t', action='store_true', help='Use distilled transformer')
    args = parser.parse_args()
    eval_folders = os.listdir("eval")
    if len(eval_folders) == 1:
        eval_folder = eval_folders[0]
    else:
        eval_folder = input("Evaluate files in folder: ").strip()
    eval_directory = os.path.join("eval", eval_folder)
    if not os.path.isdir(eval_directory):
        print(f"Directory '{eval_directory}' does not exist.")
        exit(1)
    start_time = time.time()
    evaluate_accuracy(eval_directory, args.t)
    elapsed_time = time.time() - start_time
    print(f"Evaluation completed in {elapsed_time:.2f} seconds.")

