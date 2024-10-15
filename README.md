## SLTM-Classifier

This is an learning project meant to get familiar with machine learning programming. The script can train a neural net to classify chunks of audio into content types with the help of machine learning. It is a rewrite of [amsterg](https://github.com/amsterg/Podcast-Ad-Detection)'s project (to which I concede the creative credit). The project uses an LSTM (Long Short-Term Memory) architecture to extract and process MFCC features.\
This version is optimized for home computers. It is less fine-grained and uses lower qualities than the original. Training is fast and can be done in non-painful time.
My tests show 100% accuracy on unseen data for 10 second chunks after just 2 training epochs on only five input episodes.\
This is a hobby project. The functions are set up specific to my needs and include inflexible hardcoded values fitted to my particular audio files. Use would require manual setting up and adaptation to your input structure.

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

4. **Batch Conversion:**
   - Run `all_files.py` to convert all files in the `convert` folder.

### Requirements

- Python 3
- torch
- torchvision
- librosa
- numpy
- pydub
- Others I might have forgotten (you will knwo from the errors)

Dependencies can be installed via:
  ```bash
  pip install -r requirements.txt
  ```

Feature wishlist:\
- Optimization and more optimization
- Directly feeding MFCC data instead of cutting audio files
