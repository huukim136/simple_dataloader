import os

# Dataset
dataset = "LJSpeech-1.1"
data_path = "LJSpeech-1.1"


# Optimizer
batch_size = 16

training_files = "/store/hk/hispeech2/preprocessed_kh/LJSpeech-1.1/train.txt"
validation_files = "/store/hk/hispeech2/preprocessed_kh/LJSpeech-1.1/val.txt"
text_cleaners = ["english_cleaners2"]
max_wav_value =  32768.0
sampling_rate = 22050
filter_length =  1024
hop_length =  256
win_length = 1024
n_mel_channels =  80
mel_fmin = 0.0
mel_fmax = 8000.0
# add_blank = TRUE
n_speakers = 0
# cleaned_text = true