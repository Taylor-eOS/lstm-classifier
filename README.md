## LSTM-Classifier

This is an learning project meant to get familiar with machine learning and neural network programming. The script uses a LSTM (Long Short-Term Memory) architecture to classify chunks of audio into two content types. It is a complete rewrite of a similar project from [amsterg](https://github.com/amsterg/Podcast-Ad-Detection), to which I concede the creative credit. The project extracts and process MFCC features from audio files in order to learn distinctions and be able to predict what type a unseen chunk of audio is.\
This version is optimized for home computers. It is less fine-grained and uses lower qualities than the original. Training is fast and can be done in non-painful time with a normal computer. My tests show 100% accuracy on unseen data after just 2 training epochs on only five input episodes. Inference is still slower than would be convenient.<p>
Note: This is a hobby project. It consists of crude source code that is not set up to be user friendly. The functions are made specific to my needs and include inflexible hardcoded values fitted to my particular use case. If you wanted to make use of this, it would require understanding the source code and manually setting up and adapting the code for your content.\
This project is an excellent simple example of neural net technology in use, that can be used for studying the technology.

### How to Use

1. **Training the Model:**
   - Prepare your dataset with two folders named `train` (and `val`), in which `wav` files are placed in subfolders named `A` and `B`.
   - My abandoned original [project](https://github.com/Taylor-eOS/dual-model-classifier) contains a tool for manual labeling of training data in the required format which will be read by the algorithm in order to create training segments.
   - These split points go into the file `segments.txt` in the `input` folder. Use `cut_segments_tool.py` to split audio files according to the labels in the file. This cutting would have to be adapted to your files, as it expects 9 segments.
   - Use `chunk_cutting_tool.py` to segment the input files into small chunks.
   - `split_validation_tool.py` can randomly set aside 10% of your files as a validation set. Move the files into the right folders.
   - Run the training script:
     ```bash
     python main.py --mode train
     ```
   - The model will be trained and saved.

2. **Running Inference:**
   - To classify a single audio chunk for testing functionality, use the inference mode:
     ```bash
     python main.py --mode infer --file file.wav
     ```
   - The output will indicate whether the file is classified as `A` or `B`.

3. **Batch Inference for Evaluation:**
   - To evaluate multiple files at once and get a percentage accuracy value, place your `.wav` files in the `eval/` directory with an `_a` or `_b` at the end of the file name. Run:
     ```bash
     python evaluate.py
     ```
   - The script will run through all inferences, compare the results to the labels in the file, and output the accuracy of the classification for all files in the evaluation set. You can use this to optimize the model parameters.

4. Other functionalities, I will not mention here.

### Requirements
- Python 3
- torch
- torchvision
- librosa
- numpy
- pydub
- Others I might have forgotten (you will know from the errors)

Dependencies can be installed via:
  ```bash
  pip install -r requirements.txt
  ```

Feature wishlist:<p>
- Optimization
- Directly feeding MFCC data instead of cutting audio files
- Distilling a transformer from the LSTM
- Synthesizing ground data
