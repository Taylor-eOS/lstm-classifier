import os
import subprocess
import argparse

def run_inference(file_path):
    result = subprocess.run(['python', 'main.py', '--mode', 'infer', '--file', file_path], capture_output=True, text=True)
    output = result.stdout.strip()
    if 'ads' in output:
        return 'a'
    elif 'broadcast' in output:
        return 'b'
    return None

def evaluate_accuracy(directory):
    total_files = 0
    correct_predictions = 0
    for file_name in os.listdir(directory):
        if file_name.endswith('.wav'):
            print(f'{file_name}')
            total_files += 1
            true_class = file_name.split('_')[-1][0]
            #print(f'{true_class}')
            file_path = os.path.join(directory, file_name)
            predicted_class = run_inference(file_path)
            #print(f'{predicted_class}')
            if predicted_class == true_class:
                correct_predictions += 1
                print('correct')
            else:
                print('wrong')
    if total_files > 0:
        accuracy = (correct_predictions / total_files) * 100
        print(f'Accuracy: {accuracy:.2f}% ({correct_predictions}/{total_files} correct predictions)')
    else:
        print("No .wav files found in the directory.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch inference script")
    #parser.add_argument('--dir', type=str, required=True, help="Directory containing .wav files for inference")
    #args = parser.parse_args()
    evaluate_accuracy('eval/')

