import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_io as tfio

TRAIN_RATIO = 0.6
VALIDATE_RATIO = 0.2
TEST_RATIO = 0.2

SHUFFLE_SEED = 42
BATCH_SIZE = 32
EPOCHS = 20

MODEL_NAME = "NSSD"


# TODO: convert 24-bit audio to 16-bit audio
@tf.function
def load_wav_16k_mono(path):
    file_contents = tf.io.read_file(path)
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
    return wav


# audio data is split into 0.48 second frames, thus, rf.repeat is needed to
# copy the column for each frame
def extract_embedding(yamnet_model, audio_data, variant):
    scores, embeddings, spectrogram = yamnet_model(audio_data)
    num_embeddings = tf.shape(embeddings)[0]
    return (embeddings, tf.repeat(variant, num_embeddings))


def load_yamnet():
    return hub.load("https://tfhub.dev/google/yamnet/1")


def split_dataframe_field(dataframe, field: str, splits: list[float]):
    variants = dataframe[field].unique()

    subdataframes = []
    for _ in splits:
        subdataframes.append(pd.DataFrame())

    for variant in variants:
        subset = dataframe.loc[dataframe[field] == variant].reset_index()

        start = 0
        for i, ratio in enumerate(splits):
            subset_size = int(ratio * len(subset))
            end = start + subset_size
            print(start, end)

            subdataframes[i] = pd.concat([subdataframes[i], subset.iloc[start:end]])
            start += subset_size

    # make each dataframe start at index 1
    for i, dataframe in enumerate(subdataframes):
        subdataframes[i] = dataframe.reset_index()

    return subdataframes


# splits dataframe into train, validate, test evenly based on variant
def split_dataframe(dataframe):
    dataframe = dataframe.sample(frac=1, random_state=SHUFFLE_SEED)
    return split_dataframe_field(
        dataframe, "variant", [TRAIN_RATIO, VALIDATE_RATIO, TEST_RATIO]
    )


# takes in pandas dataframe and relevant fields, outputs tensorflow dataset
def preprocess_dataframe(yamnet_model, dataframe, class_to_id):
    # map classes to their ids
    dataframe["variant"] = dataframe["variant"].map(
        lambda variant: class_to_id[variant]
    )
    # make tensorflow dataset with relevant fields
    dataset = tf.data.Dataset.from_tensor_slices(
        (dataframe["path"], dataframe["variant"])
    )
    # convert audio to 16k mono
    dataset = dataset.map(lambda path, variant: (load_wav_16k_mono(path), variant))
    # applies the embedding extraction model to wav data
    dataset = dataset.map(
        lambda audio_data, variant: extract_embedding(yamnet_model, audio_data, variant)
    ).unbatch()
    # cache, batch, and prefetch dataset
    dataset = dataset.cache().batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return dataset


def train(train, validate, num_classes):
    # TODO: add input layers for extra fields
    model = tf.keras.Sequential(
        [
            # embeddings
            tf.keras.layers.Input(
                shape=(1024), dtype=tf.float32, name="input_embedding"
            ),
            # hidden layer
            tf.keras.layers.Dense(512, activation="relu"),
            # output
            tf.keras.layers.Dense(num_classes),
        ],
        name=MODEL_NAME,
    )
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer="adam",
        metrics=["accuracy"],
    )
    model.fit(
        train,
        epochs=EPOCHS,
        validation_data=validate,
        callbacks=tf.keras.callbacks.EarlyStopping(
            monitor="loss", patience=3, restore_best_weights=True
        ),
    )
    return model


class ReduceMeanLayer(tf.keras.layers.Layer):
    def __init__(self, axis=0, **kwargs):
        super(ReduceMeanLayer, self).__init__(**kwargs)
        self.axis = axis

    def call(self, input):
        return tf.math.reduce_mean(input, axis=self.axis)


# saves the model with a simple wav input layer
def save_simple(yamnet_model, model, save_path):
    input_segment = tf.keras.layers.Input(shape=(), dtype=tf.float32, name="audio")
    embedding_extraction_layer = hub.KerasLayer(
        yamnet_model, trainable=False, name="yamnet"
    )
    _, embeddings_output, _ = embedding_extraction_layer(input_segment)
    serving_outputs = model(embeddings_output)
    serving_outputs = ReduceMeanLayer(axis=0, name="classifier")(serving_outputs)
    serving_model = tf.keras.Model(input_segment, serving_outputs)
    serving_model.save(save_path)
