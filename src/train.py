import os

from IPython import display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_io as tfio

import download_manual  # noqa: F401

yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")

pd_data = pd.read_csv(download_manual.CSV_PATH)
map_class_to_id = {k: i for i, k in enumerate(pd_data["positive_labels"].unique())}

filtered_pd = pd_data
filtered_pd["target"] = filtered_pd["positive_labels"].apply(
    lambda name: map_class_to_id[name]
)
filtered_pd["filename"] = filtered_pd.apply(
    lambda row: download_manual.OUTPUT_DIR
    / download_manual.format_path(
        row["YTID"], row["start_seconds"], row["end_seconds"]
    ),
    axis=1,
)

print(filtered_pd.head())

# TODO: fold
filenames = filtered_pd["filename"]
targets = filtered_pd["target"]
folds = filtered_pd["fold"]

main_ds = tf.data.Dataset.from_tensor_slices((filenames, targets, folds))


@tf.function
def load_wav_16k_mono(filename):
    """Load a WAV file, convert it to a float tensor, resample to 16 kHz single-channel audio."""
    file_contents = tf.io.read_file(filename)
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
    return wav


def load_wav_for_map(filename, label, fold):
    return load_wav_16k_mono(filename), label, fold


main_ds = main_ds.map(load_wav_for_map)


# applies the embedding extraction model to a wav data
def extract_embedding(wav_data, label, fold):
    """run YAMNet to extract embedding from the wav data"""
    scores, embeddings, spectrogram = yamnet_model(wav_data)
    num_embeddings = tf.shape(embeddings)[0]
    return (
        embeddings,
        tf.repeat(label, num_embeddings),
        tf.repeat(fold, num_embeddings),
    )


# extract embedding
main_ds = main_ds.map(extract_embedding).unbatch()
