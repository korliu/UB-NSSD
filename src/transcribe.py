import argparse
import os

import pandas as pd
import preprocess
import result
import tensorflow as tf
import training
import utils
import visualize
import keras
from pathlib import Path


def parse_output(prediction):
    class_dict = utils.c2i(Path("src", "classes.csv"))
    classes = list(class_dict.keys())

    chunk_size = len(classes)

    for s in range(0, len(prediction), chunk_size):
        pred = prediction[s : (s + chunk_size)]

        top_class = tf.math.argmax(pred)
        class_probabilities = tf.nn.softmax(pred, axis=-1)

        # print(top_class, classes[top_class], class_probabilities)

    class_probabilities = tf.nn.softmax(prediction, axis=-1)

    print(class_probabilities, class_probabilities[0].numpy())
    pass


MODEL_DIR = "models"

STORE = "store"
STORE_CONST = "store_const"


parser = argparse.ArgumentParser(
    description="UB-NSSD YAMNet transfer learning model to transcribe audio (only-intake)"
)

parser.add_argument("audio_path", help="Audio file to transcribe")

# can get the sr from audio, so unsure if necessary
# parser.add_argument("--sr", action="store_const")

# window size
parser.add_argument(
    "--size",
    nargs="?",
    const=1,
    type=float,
    default=0.98,
    help="Size of prediction window, in seconds, default=0.98",
)
# step size
parser.add_argument(
    "--step",
    nargs="?",
    const=1,
    type=float,
    default=0.10,
    help="Step to move window, in seconds, default=0.10",
)

parser.add_argument(
    "--save_path", action=STORE, help="File location to store the prediction graph"
)

parser.add_argument(
    "--model_version",
    nargs="?",
    const=1,
    default="only_intake",
    help="Version of the model to transcribe with. \
                        [only_intake, all], default=only_intake",
)


args = parser.parse_args()

model_path = Path("models", args.model_version)
model = tf.keras.saving.load_model(model_path)

model.summary()


def split_audio_segments(wav, segment_duration=0.96, overlap=0):
    segment_samples = int(segment_duration * 16000)
    overlap_samples = int(overlap * 16000)

    num_segments = int(
        (len(wav) - overlap_samples) / (segment_samples - overlap_samples)
    )

    segments = []
    for i in range(num_segments):
        start = i * (segment_samples - overlap_samples)
        end = start + segment_samples
        segment = wav[start:end]
        segments.append(segment)

    return segments


audio = training.load_wav_16k_mono(args.audio_path)
segments = split_audio_segments(audio)
for segment in segments[0:1]:
    # if the segment size is 0.96s this should only have 3 elements
    prediction = model(segment)
    print(prediction)

    top_class = tf.math.argmax(prediction)
    class_probabilities = tf.nn.softmax(prediction, axis=-1)
    print(top_class, class_probabilities)
