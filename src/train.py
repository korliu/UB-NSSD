import os

from IPython import display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_io as tfio

import download_manual  # noqa: F401

# 80% train, 20% validate/test (for now)
TRAIN_RATIO = 0.8

yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")

pd_data = pd.read_csv(download_manual.CSV_PATH)
map_class_to_id = {k: i for i, k in enumerate(pd_data["positive_labels"].unique())}
my_classes = list(map_class_to_id.keys())

filtered_pd = pd_data
filtered_pd["target"] = filtered_pd["positive_labels"].apply(
    lambda name: map_class_to_id[name]
)
filtered_pd["filename"] = filtered_pd.apply(
    lambda row: str(
        download_manual.OUTPUT_DIR
        / download_manual.format_path(
            row["YTID"], row["start_seconds"], row["end_seconds"]
        )
    ),
    axis=1,
)

train = filtered_pd.sample(frac=TRAIN_RATIO)
test = filtered_pd.drop(train.index)
train["fold"] = 1
test["fold"] = 2

filtered_pd = pd.concat([train, test])
print(filtered_pd.head())

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

cached_ds = main_ds.cache()
train_ds = cached_ds.filter(lambda embedding, label, fold: fold < 2)
val_ds = cached_ds.filter(lambda embedding, label, fold: fold == 2)
test_ds = cached_ds.filter(lambda embedding, label, fold: fold == 2)


# remove the folds column now that it's not needed anymore
def remove_fold_column(embedding, label, fold):
    return embedding, label


train_ds = train_ds.map(remove_fold_column)
val_ds = val_ds.map(remove_fold_column)
test_ds = test_ds.map(remove_fold_column)

train_ds = train_ds.cache().shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.cache().batch(32).prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.cache().batch(32).prefetch(tf.data.AUTOTUNE)
my_model = tf.keras.Sequential(
    [
        tf.keras.layers.Input(shape=(1024), dtype=tf.float32, name="input_embedding"),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dense(len(my_classes)),
    ],
    name="my_model",
)

my_model.summary()

my_model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer="adam",
    metrics=["accuracy"],
)

callback = tf.keras.callbacks.EarlyStopping(
    monitor="loss", patience=3, restore_best_weights=True
)

history = my_model.fit(train_ds, epochs=20, validation_data=val_ds, callbacks=callback)

loss, accuracy = my_model.evaluate(test_ds)

print("Loss: ", loss)
print("Accuracy: ", accuracy)


# TODO: only runs the first one for now
# testing_wav_data = load_wav_16k_mono(test.iloc[0]["filename"])
# testing_wav_data = load_wav_16k_mono("datasets/test.wav")

# scores, embeddings, spectrogram = yamnet_model(testing_wav_data)
# result = my_model(embeddings).numpy()

# inferred_class = my_classes[result.mean(axis=0).argmax()]
# print(f"The main sound is: {inferred_class}")


class ReduceMeanLayer(tf.keras.layers.Layer):
    def __init__(self, axis=0, **kwargs):
        super(ReduceMeanLayer, self).__init__(**kwargs)
        self.axis = axis

    def call(self, input):
        return tf.math.reduce_mean(input, axis=self.axis)


saved_model_path = "./swallow_yamnet"

input_segment = tf.keras.layers.Input(shape=(), dtype=tf.float32, name="audio")
embedding_extraction_layer = hub.KerasLayer(
    "https://tfhub.dev/google/yamnet/1", trainable=False, name="yamnet"
)
_, embeddings_output, _ = embedding_extraction_layer(input_segment)
serving_outputs = my_model(embeddings_output)
serving_outputs = ReduceMeanLayer(axis=0, name="classifier")(serving_outputs)
serving_model = tf.keras.Model(input_segment, serving_outputs)
serving_model.save(saved_model_path, include_optimizer=False)
