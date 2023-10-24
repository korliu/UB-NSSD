import argparse

import pandas as pd
import tensorflow as tf

import train

DATASET_PATH = "datasets/all_data.csv"
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

    y_true, y_pred = [], []
    for row in test_split:
        waveform = train.load_wav_16k_mono(row[0])
        results = model(waveform)
        top_class = tf.math.argmax(results)
        inferred_class = classes[top_class]
        class_probabilities = tf.nn.softmax(results, axis=-1)
        top_score = class_probabilities[top_class]

        y_true.append(class_to_id[row[1].numpy().decode()])
        y_pred.append(top_class.numpy())

        print(
            f"actual: {row[1].numpy().decode()}, inferred: {inferred_class}, score: {top_score.numpy()}",
        )

    auc = tf.keras.metrics.AUC()
    auc.update_state(y_true, y_pred)
    print(f"AUC: {auc.result().numpy()}")

    precision = tf.keras.metrics.Precision()
    precision.update_state(y_true, y_pred)
    precision.result().numpy()
    print(f"Precision: {precision.result().numpy()}")

    recall = tf.keras.metrics.Recall()
    recall.update_state(y_true, y_pred)
    recall.result().numpy()
    print(f"Recall: {recall.result().numpy()}")

    # f1 = tf.keras.metrics.F1Score()
    # f1.update_state(y_true, y_pred)
    # f1.result().numpy()
    # print(f"F1: {f1.result().numpy()}")
