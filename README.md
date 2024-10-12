## SLTM-Classifier

This is an educational machine learning project that can classify audio chunks into types with the help of a machine learning.
It is a rewrite of [podcast-ad-detection](https://github.com/amsterg/Podcast-Ad-Detection), to which the creative credit belongs.<p>
The project uses an LSTM (Long Short-Term Memory) architecture to process MFCC (Mel Frequency Cepstral Coefficient) features extracted from audio files.
Recent tests show 100% accuracy on unseen data for 10 second chunks after just 2 training epochs on only five input episodes.
This projects' functions are set up fairly specific to my needs and include hardcoded values. Use would require manual setting up.<p>
Some tools for splitting up audio into smaller chunks are provided, and have to be run in the right order.<p>
Currently no functionality for cutting fiels is provided.

### How to Use

1. **Training the Model:**
   - Prepare your dataset with two folders named `A` and `B`, where `A` contains `wav` files of one type and `B` of the other.
   - Run the training script:
     ```bash
     python main.py --mode train
     ```
   - The model will be trained and saved.

2. **Running Inference:**
   - To classify a single audio file, use the inference mode:
     ```bash
     python main.py --mode infer --file file.wav
     ```
   - The output will indicate whether the file is classified as `A` (ads) or `B` (broadcast).

3. **Batch Inference for Evaluation:**
   - To evaluate multiple files at once, place your `.wav` files in the `eval/` directory and run:
     ```bash
     python evaluate.py
     ```
   - The script will output the accuracy of the classification across all files in the evaluation set.

### Requirements

- Python 3.x
- Torch
- Librosa
- Any other necessary dependencies can be installed via:
  ```bash
  pip install -r requirements.txt
  ```
