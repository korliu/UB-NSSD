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