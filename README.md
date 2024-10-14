## SLTM-Classifier

This is an learning project that trains a neural net to classify chunks of audio into content types with the help of machine learning.
The project uses an LSTM (Long Short-Term Memory) architecture to process MFCC (Mel Frequency Cepstral Coefficient) features extracted from audio files.
It is a rewrite of [amsterg](https://github.com/amsterg/Podcast-Ad-Detection)'s project, to which the creative credit belongs.
This version is optimized for home computers. It is less fine-grained and uses lower qualities. Training is fast and can be done in non-painful time.
My tests show 100% accuracy on unseen data for 10 second chunks after just 2 training epochs on only five input episodes.<p>
This projects' functions are set up fairly specific to my needs and include hardcoded values. Use would require manual setting up and adaptation to your input structure.
Some tools for splitting up training data into smaller chunks are provided, and have to be run in the right order.

### How to Use

1. **Training the Model:**
   - Prepare your dataset with two folders named `A` and `B`, where `A` contains `wav` files of one type and `B` of the other.
   - My abandoned original  [project](https://github.com/Taylor-eOS/dual-model-classifier) contains a tool for manual labeling of training data in the required format and an example of the format used in the files.
   - Use `cut_segments_tool.py` to split audio files according to the labels in a `segments.txt` file. This would have to be adapted to your input. That project also contains a function to automatically converts mp3 files to wav.
   - Use `chunk_cutting_tool.py` to segment the input files into small chunks.
   - `split_validation_tool.py` can automatically set aside 10% of your files as a validation set. Move the files into folders `A` and `B` in the `train` and `val` folders.
   - Run the training script:
     ```bash
     python main.py --mode train
     ```
   - The model will be trained and saved.

2. **Running Inference:**
   - To classify a single audio chunk, use the inference mode:
     ```bash
     python main.py --mode infer --file file.wav
     ```
   - The output will indicate whether the file is classified as `A` or `B`.

3. **Batch Inference for Evaluation:**
   - To evaluate multiple files at once and get a percentage accuracy value, place your `.wav` files in the `eval/` directory with an `a` or `b` in the file name last after an underscore and run:
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
Requirements may not be up to date.
