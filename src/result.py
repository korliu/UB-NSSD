import os
import csv

from IPython import display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_io as tfio

import download_manual

OUTPUT = "outputs/yamnet_analysis.csv"

yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
reloaded_model = tf.saved_model.load("./swallow_yamnet")

pd_data = pd.read_csv(download_manual.CSV_PATH)
map_class_to_id = {k: i for i, k in enumerate(pd_data["positive_labels"].unique())}
my_classes = list(map_class_to_id.keys())


@tf.function
def load_wav_16k_mono(filename):
    """Load a WAV file, convert it to a float tensor, resample to 16 kHz single-channel audio."""
    file_contents = tf.io.read_file(filename)
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
    return wav


class_map_path = yamnet_model.class_map_path().numpy().decode("utf-8")
class_names = list(pd.read_csv(class_map_path)["display_name"])

pd_data["filename"] = pd_data.apply(
    lambda row: str(
        download_manual.OUTPUT_DIR
        / download_manual.format_path(
            row["YTID"], row["start_seconds"], row["end_seconds"]
        )
    ),
    axis=1,
)

results = [["youtube_id", "food", "start", "end", "correct", "inferred", "probability"]]
for _, row in pd_data.iterrows():
    waveform = load_wav_16k_mono(row["filename"])
    reloaded_results = reloaded_model(waveform)
    your_top_class = tf.math.argmax(reloaded_results)
    your_inferred_class = my_classes[your_top_class]
    class_probabilities = tf.nn.softmax(reloaded_results, axis=-1)
    your_top_score = class_probabilities[your_top_class]
    results.append(
        [
            row["YTID"],
            row["food"],
            row["start_seconds"],
            row["end_seconds"],
            row["positive_labels"],
            your_inferred_class,
            str({your_top_score}),
        ]
    )
    # print(f"[Your model] The main sound is: {your_inferred_class} ({your_top_score})")

with open(OUTPUT, "w") as f:
    writer = csv.writer(f)
    writer.writerows(results)
