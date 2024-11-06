## LSTM-Classifier

This was a exploration project meant to get familiar with machine learning and neural network programming. The script uses a LSTM (Long Short-Term Memory) architecture to classify chunks of audio into two content types. The project extracts and process MFCC features from audio files in order to train a neural net to learn the distinctive patterns that distinguish the types of input data. It can then predict what type a unseen chunk of audio is. It is a complete reassembly of a project from [amsterg](https://github.com/amsterg/Podcast-Ad-Detection), to whom I concede the creative credit.\
This version is optimized for home computers. It is less fine-grained and uses lower qualities than the original. Training is fast and can be done in non-painful time on a CPU. Most of the time it takes is actually to convert the files to wav, since the MFCC extraction had problems with mp3 input. My tests show near-100% accurate predictions on unseen data after just a few training epochs and with only five input episodes.\
The project contains tools for labeling ground truth data, but these require some time investment to be used correctly.\
Note: This is a hobby project. It consists of crude source code that is not set up to be user friendly. The functions are made specific to a particular use case and include hardcoded values that may not work with all inputs. The script may at times not check for deviations from the expected use. To get this to work, it would require understanding the source code and manually setting up and adapting the code for your content.\
This project is an excellent simple example of neural net technology in use, that can be used for studying the technology.

### Instructions

1. **Creating ground truth:**
   - Put your unsegmented wav files and a space-separated `segments.txt` containing the ground truth timestamps into the `input` folder. These labels can be created using `tool_manually_label_audio.py`.
   - Run `tool_cut_segments.py` to split your files up according to the labels.
   - Run `tool_separate_chunks.py` to separate your files into training-sized chunks. The script will randomly set aside a given proportion of your files as a validation set and move the files into the right folders.
   - Optionally you can use `tool_shuffle_more_files` to synthesize more files to even out the amount of `A` and `B` types.
   
2. **Training the Model:**
   - Run the training script:
     ```bash
     python main.py
     ```
   - If everything was placed correctly, the model will be trained and saved.

3. **Running Inference:**
   - To classify a single audio chunk for testing functionality, use the inference mode:
     ```bash
     python main.py file.wav
     ```
   - The output will indicate whether the file is classified as `A` or `B`.

4. **Batch Inference for Evaluation:**
   - To evaluate multiple chunks at once and get a percentage accuracy value for model evaluation, place your wav files in the `eval` directory with an `_a` or `_b` at the end of the file name. Run:
     ```bash
     python tool_evaluate.py
     ```
   - The script will run through all inferences, compare the results to the labels in the file, and output the accuracy of the classification for all files in the evaluation set. You can use this to optimize the model parameters.

Transformer distillation currently doesn't work because of dimension mismatches. But there is a file to train a transformer from ground truth.

### Requirements
- Python 3
- torch
- torchvision
- librosa
- numpy
- pydub
- pygame (tools only)
- Others I might have forgotten (you will know from the errors)

Dependencies can be installed via:
  ```bash
  pip install -r requirements.txt
  ```
