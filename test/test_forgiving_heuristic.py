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
    from process_file import apply_forgiving_heuristic
    parser = argparse.ArgumentParser(description="Apply forgiving heuristic to predictions from a file.")
    parser.add_argument('--file', type=str, default='list.txt', help='Path to the predictions file (default: list.txt)')
    args = parser.parse_args()
    predictions = read_predictions(args.file)
    if not predictions:
        print("No valid predictions found. Exiting.")
        sys.exit(0)
    print(f"Loaded {len(predictions)} predictions from '{args.file}'.")
    corrected_predictions = apply_forgiving_heuristic(predictions)
    print("Corrected Predictions:")
    for idx, pred in enumerate(corrected_predictions, start=1):
        print(f"Chunk {idx}: {pred}")

if __name__ == "__main__":
    main()
