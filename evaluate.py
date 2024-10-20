import os
import re
import time
import argparse
from concurrent.futures import ThreadPoolExecutor
from main import main as main_main
from distill_transformer import main_transformer as distill_main

def run_inference(file_path):
    #print('lstm')
    pred_class, prob_B, logits = main_main(mode='infer', file=file_path)
    return pred_class, prob_B, logits

def run_inference_transformer(file_path):
    #print('transformer')
    pred_class, prob_B, logits = distill_main(mode='infer', file=file_path)
    return pred_class, prob_B, logits

def process_file(file_name, directory, use_transformer):
    file_path = os.path.join(directory, file_name)
    true_class = file_name.split('_')[-1][0].upper() #Ensure it's uppercase for comparison
    if use_transformer:
        pred_class, _, _ = run_inference_transformer(file_path)
    else:
        pred_class, _, _ = run_inference(file_path)
    pred_class = pred_class.upper() #Ensure consistency in comparison
    file_number_match = re.search(r'\d+', file_name)
    file_number = file_number_match.group() if file_number_match else "N/A"
    if pred_class == true_class:
        print(f'{file_number:2} {pred_class} correct')
        return True
    else:
        print(f'{file_number} {pred_class} wrong')
        return False

def evaluate_accuracy(directory, use_transformer):
    #print(f'{use_transformer}')
    files = [file_name for file_name in os.listdir(directory) if file_name.endswith('.wav')]
    if not files:
        print("No .wav files found in the directory.")
        return
    total_files = len(files)
    correct_predictions = 0
    with ThreadPoolExecutor() as executor:
        results = executor.map(lambda file_name: process_file(file_name, directory, use_transformer), files)
        correct_predictions = sum(results)
    accuracy = (correct_predictions / total_files) * 100
    print(f'Accuracy: {accuracy:.2f}% ({correct_predictions}/{total_files} correct predictions)')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch inference script")
    parser.add_argument('--transformer', action='store_true', help='Use distilled transformer model?')
    args = parser.parse_args()
    eval_folder = input("Evaluate files in folder: ").strip()
    eval_directory = os.path.join("eval", eval_folder)
    if not os.path.isdir(eval_directory):
        print(f"Directory '{eval_directory}' does not exist.")
        exit(1)
    start_time = time.time()
    evaluate_accuracy(eval_directory, args.transformer)
    elapsed_time = time.time() - start_time
    print(f"Evaluation completed in {elapsed_time:.2f} seconds.")

