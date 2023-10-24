import argparse

import pandas as pd
import tensorflow as tf

import train

DATASET_PATH = "datasets/foods_data.csv"
SAVE_PATH = "nssd_model"

parser = argparse.ArgumentParser(description="UB-NSSD YAMNet transfer learning model")
parser.add_argument("--train", action="store_true", help="Whether to train the model")
parser.add_argument("--test", action="store_true", help="Whether to test the model")
args = parser.parse_args()

if args.train:
    yamnet_model = train.load_yamnet()

    dataframe = pd.read_csv(DATASET_PATH)
    class_to_id = {k: i for i, k in enumerate(dataframe["variant"].unique())}
    classes = list(class_to_id.keys())

    dataset = train.preprocess_dataframe(yamnet_model, dataframe, class_to_id)
    train_split, validate_split, test_split = train.split_dataset(
        dataset, len(dataframe)
    )

    model = train.train(train_split, validate_split, len(classes))
    train.save_simple(yamnet_model, model, SAVE_PATH)


if args.test:
    dataframe = pd.read_csv(DATASET_PATH)
    _, _, test_split = train.split_dataframe(dataframe, len(dataframe))

    class_to_id = {k: i for i, k in enumerate(dataframe["variant"].unique())}
    classes = list(class_to_id.keys())

    # model = tf.saved_model.load(SAVE_PATH)
    model = tf.keras.models.load_model(SAVE_PATH)
    model.summary()
    # model.compile(metrics=["accuracy"])
    for row in test_split:
        waveform = train.load_wav_16k_mono(row[0])
        results = model(waveform)
        top_class = tf.math.argmax(results)
        inferred_class = classes[top_class]
        class_probabilities = tf.nn.softmax(results, axis=-1)
        top_score = class_probabilities[top_class]
        print(
            f"actual: {row[1].numpy().decode()}, inferred: {inferred_class}, score: {top_score.numpy()}"
        )
