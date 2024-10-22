## LSTM-Classifier

This was a exploration project meant to get familiar with machine learning and neural network programming. The script uses a LSTM (Long Short-Term Memory) architecture to classify chunks of audio into two content types. The project extracts and process MFCC features from audio files in order to train a neural net to learn the distinctive patterns that distinguish the types of input data. It can then predict what type a unseen chunk of audio is. It is a complete reassembly of a project from [amsterg](https://github.com/amsterg/Podcast-Ad-Detection), to whom I concede the creative credit.\
This version is optimized for home computers. It is less fine-grained and uses lower qualities than the original. Training is fast and can be done in non-painful time on a CPU. My tests show near-100% accurate predictions on unseen data after just a few training epochs and with only five input episodes. Inference is still slower than would be convenient.<p>
Note: This is a hobby project. It consists of crude source code that is not set up to be user friendly. The functions are made specific to my needs and include hardcoded values fitted to my particular use case. The script assumes that neccesary files and folders exist without creating them if they don't. If you want to make use of this, it would require understanding the source code and manually setting up and adapting the code for your content.\
This project is an excellent simple example of neural net technology in use, that can be used for studying the technology.

### How to Use

1. **Training the Model:**
   - Place your unsegmented training data in the folder `input` along with a `segments.txt` file that contains the timestamps between segments.
   - My other [project](https://github.com/Taylor-eOS/dual-model-classifier) contains a tool for manually labeling training data in the required format, which will be read by the algorithm in order to create training segments.
   - Use `cut_segments_tool.py` to split your files according to the labels in the file. This cutting would have to be adapted to your files, as it expects a format.
   - Use `separate_chunks_tool.py` to segment the input files into small chunks for training. It will randomly set aside 10% of your files as a validation set and move the files into the right folders.
   - Run the training script:
     ```bash
     python main.py
     ```
   - If everything was placed correctly, the model will be trained and saved.

2. **Running Inference:**
   - To classify a single audio chunk for testing functionality, use the inference mode:
     ```bash
     python main.py --f file.wav
     ```
   - The output will indicate whether the file is classified as `A` or `B`.

3. **Batch Inference for Evaluation:**
   - To evaluate multiple chunks at once and get a percentage accuracy value for model evaluation, place your `.wav` files in the `eval` directory with an `_a` or `_b` at the end of the file name. Run:
     ```bash
     python evaluate.py
     ```
   - The script will run through all inferences, compare the results to the labels in the file, and output the accuracy of the classification for all files in the evaluation set. You can use this to optimize the model parameters.

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
