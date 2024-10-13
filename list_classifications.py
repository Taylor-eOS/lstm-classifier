import os
import subprocess
import argparse

def run_inference(file_path):
    result = subprocess.run(['python', 'main.py', '--mode', 'infer', '--file', file_path], capture_output=True, text=True)
    output = result.stdout.strip()
    print(f'o: {output}')
    if 'A' in output:
        return 'A'
    elif 'B' in output:
        return 'B'
    print('Error: No inference made')
    return None

def evaluate_accuracy(directory):
    i = 1
    with open('output.txt', 'a') as f:
        #for file_name in os.listdir(directory):
        file_names = sorted(os.listdir(directory))
        for file_name in file_names:
            if file_name.endswith('.wav'):
                file_path = os.path.join(directory, file_name)
                predicted_class = run_inference(file_path)
                result = f'{i} {file_name}: {predicted_class}'
                print(result)
                f.write(result + '\n')
                i += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="List Classifications")
    evaluate_accuracy('classify/')

