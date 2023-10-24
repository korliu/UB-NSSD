import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_io as tfio

TRAIN_RATIO = 0.6
VALIDATE_RATIO = 0.2
TEST_RATIO = 1 - (TRAIN_RATIO + VALIDATE_RATIO)

SHUFFLE_SEED = 42
BATCH_SIZE = 32
EPOCHS = 20

MODEL_NAME = "NSSD"


@tf.function
def load_wav_16k_mono(path):
    file_contents = tf.io.read_file(path)
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
    return wav


def extract_embedding(yamnet_model, audio_data, variant):
    scores, embeddings, spectrogram = yamnet_model(audio_data)
    num_embeddings = tf.shape(embeddings)[0]
    return (embeddings, tf.repeat(variant, num_embeddings))


def load_yamnet():
    return hub.load("https://tfhub.dev/google/yamnet/1")


# takes in pandas dataframe and relevant fields, outputs tensorflow dataset
def preprocess_dataframe(yamnet_model, dataframe, class_to_id):
    # make tensorflow dataset with relevant fields
    dataset = tf.data.Dataset.from_tensor_slices(
        (dataframe["path"], dataframe["variant"])
    )
    # map classes to their ids
    dataset = dataset.map(lambda path, variant: (path, class_to_id[map]))
    # convert audio to 16k mono
    dataset = dataset.map(lambda path, variant: (load_wav_16k_mono(path), variant))
    # applies the embedding extraction model to wav data
    dataset = dataset.map(
        lambda audio_data, variant: extract_embedding(yamnet_model, audio_data, variant)
    ).unbatch()
    # cache, batch, and prefetch dataset
    dataset = dataset.cache().batch(BATCH_SIZE)

    return dataset


def split_dataframe(dataframe):
    return split_dataset(
        # TODO: keep other columns as well
        tf.data.Dataset.from_tensor_slices((dataframe["path"], dataframe["variant"]))
    )


# takes a tensorflow dataset and splits it into a train/test/validate dataset
def split_dataset(dataset):
    size = tf.data.experimental.cardinality(dataset)
    train_size = TRAIN_RATIO * size
    validate_size = VALIDATE_RATIO * size
    test_size = TEST_RATIO * size

    # TODO: need to evenly distrbute classes among splits
    dataset.shuffle(seed=SHUFFLE_SEED)
    train = (
        # TODO: add len of dataset to shuffle if errors
        dataset.take(train_size)
        .shuffle()
        .prefetch(tf.data.AUTOTUNE)
    )
    validate = dataset.skip(train_size).take(validate_size).prefetch(tf.data.AUTOTUNE)
    test = (
        dataset.skip(train_size + validate_size)
        .take(test_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    return train, validate, test


def train(train, validate, num_classes):
    # TODO: add input layers based on extra fields
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
