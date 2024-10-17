import os
import re
import time
import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor

def run_inference(file_path):
    result = subprocess.run(['python', 'main.py', '--mode', 'infer', '--file', file_path], capture_output=True, text=True)
    output = result.stdout.strip()
    if 'A' in output:
        #print('A')
        return 'a'
    elif 'B' in output:
        #print('A')
        return 'b'
    else:
        print('Error: No inference made')
        return None

def process_file(file_name, directory):
    file_path = os.path.join(directory, file_name)
    true_class = file_name.split('_')[-1][0]
    predicted_class = run_inference(file_path)
    file_number = re.search(r'\d+', file_name).group()
    if predicted_class == true_class:
        print(f'{file_number} {predicted_class} correct')
        return True
    else:
        print(f'{file_number} {predicted_class} wrong')
        return False

def evaluate_accuracy(directory):
    total_files = 0
    correct_predictions = 0
    files = [file_name for file_name in os.listdir(directory) if file_name.endswith('.wav')]
    if not files:
        print("No .wav files found in the directory.")
        return
    with ThreadPoolExecutor() as executor:
        results = executor.map(lambda file_name: process_file(file_name, directory), files)
    total_files = len(files)
    correct_predictions = sum(results)
    if total_files > 0:
        accuracy = (correct_predictions / total_files) * 100
        print(f'Accuracy: {accuracy:.2f}% ({correct_predictions}/{total_files} correct predictions)')
    else:
        print("No .wav files found in the directory.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch inference script")
    start_time = time.time()
    eval_folder = input("Evaluate files in folder: ")
    evaluate_accuracy("eval/" + eval_folder + "/")
    elapsed_time = time.time() - start_time
    print(f"Evaluation completed in {elapsed_time:.2f} seconds.")

