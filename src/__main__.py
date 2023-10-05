import matplotlib.pyplot as plt
import utils
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as tf_hub
import tensorflow_io as tfio

# TODO: make path more foolproof
# utils.get_audio_from_yt(
# "https://www.youtube.com/watch?v=LRvwtlkV-IQ", "datasets/yt.wav"
# )

# https://www.tensorflow.org/tutorials/audio/transfer_learning_audio

YAMNET_MODEL_LINK  = "https://tfhub.dev/google/yamnet/1"

yamnet_model = tf_hub.load(YAMNET_MODEL_LINK)

test_path = os.path.join("datasets","test.wav")
# test_wav_file = tf.keras.utils.get_file(test_path)

# Load the model.
model = tf_hub.load(YAMNET_MODEL_LINK)

# Input: 3 seconds of silence as mono 16 kHz waveform samples.
# waveform = np.zeros(3 * 16000, dtype=np.float32)

waveform, sr = torchaudio.load(filepath=test_path)
