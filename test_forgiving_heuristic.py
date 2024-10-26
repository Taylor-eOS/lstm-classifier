import sys

def read_predictions(file_path):
    predictions = []
    try:
        with open(file_path, 'r') as f:
            for line_number, line in enumerate(f, start=1):
                prediction = line.strip()
                if prediction not in ('A', 'B'):
                    print(f"Warning: Invalid prediction '{prediction}' on line {line_number}. Skipping.")
                    continue
                predictions.append(prediction)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while reading '{file_path}': {e}")
        sys.exit(1)
    return predictions

def main():
    import argparse
    from process_file import run_forgiving_scripts
    parser = argparse.ArgumentParser(description="Apply forgiving heuristic to predictions from a file.")
    parser.add_argument('--file', type=str, default='list.txt', help='Path to the predictions file (default: list.txt)')
    args = parser.parse_args()
    predictions = read_predictions(args.file)
    if not predictions:
        print("No valid predictions found. Exiting.")
        sys.exit(0)
    print(f"Loaded {len(predictions)} predictions from '{args.file}'.")
    predictions = run_forgiving_scripts(predictions)
    print("Corrected Predictions:")
    for idx, pred in enumerate(predictions, start=1):
        print(f"Chunk {idx}: {pred}")
    if all(pred == 'A' for pred in predictions):
        print('All predictions are A')
    elif all(pred == 'B' for pred in predictions):
        print('All predictions are B')
    else:
        print('Predictions are both types')

if __name__ == "__main__":
    main()

