import matplotlib.pyplot as plt
import utils
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as tf_hub
import tensorflow_io as tf_io
import os
from IPython import display
import torchaudio
import csv
import io
import IPython
from keras.models import load_model

# TODO: make path more foolproof
# utils.get_audio_from_yt(
# "https://www.youtube.com/watch?v=LRvwtlkV-IQ", "datasets/yt.wav"
# )

# https://www.tensorflow.org/tutorials/audio/transfer_learning_audio


YAMNET_MODEL_LINK = "https://tfhub.dev/google/yamnet/1"

yamnet_model = tf_hub.load(YAMNET_MODEL_LINK)

test_path = os.path.join("datasets","test.wav")
# test_wav_file = tf.keras.utils.get_file(test_path)

# Input: 3 seconds of silence as mono 16 kHz waveform samples.
# waveform = np.zeros(3 * 16000, dtype=np.float32)

waveform, sr = torchaudio.load(filepath=test_path)
print(waveform,sr)
transform = torchaudio.transforms.Resample(sr,16000)
waveform = transform(waveform)

np_wav = waveform.numpy()
print(np.shape(np_wav))
print(np_wav)

np_wav = np_wav[0]
print(waveform,np_wav)

_ = plt.plot(np_wav)


# plt.show()
display.Audio(np_wav, rate=16000)

class_map_path = yamnet_model.class_map_path().numpy().decode('utf-8')
print(class_map_path, "test")
print(yamnet_model)

class_names = list(pd.read_csv(class_map_path)['display_name'])

print(class_names)

scores, embeddings, spectrogram = yamnet_model(np_wav)
class_scores = tf.reduce_mean(scores, axis=0)
top_class = tf.math.argmax(class_scores)
inferred_class = class_names[top_class]

print(f'The main sound is: {inferred_class}')
print(f'The embeddings shape: {embeddings.shape}')