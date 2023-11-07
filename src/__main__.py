import argparse
import os

import result
import training
import visualize
import utils

from matplotlib import pyplot as plt
import pandas as pd
import sklearn.metrics as sk_metrics
import tensorflow as tf
from imblearn.under_sampling import RandomUnderSampler

DATASET_PATH = "datasets/all_data.csv"
MODEL_DIR = "models"
RESULT_DIR = "outputs"

BALANCE_MARGIN = 0.1  # 10%


# TODO: lots of options for over/undersampling, should check them out
def sample_dataframe(dataframe):
    return RandomUnderSampler(
        sampling_strategy="not minority", random_state=1
    ).fit_resample(dataframe, dataframe["variant"])[0]


# returns a list of dataframes to create models for
def dataframe_versions():
    dataframe = pd.read_csv(DATASET_PATH)
    dataframe = dataframe.loc[dataframe["variant"] != "other"]

    return {
        "all": sample_dataframe(dataframe),
        "only_intake": sample_dataframe(
            dataframe.loc[dataframe["source"] == "food_intake_dataset"]
        ),
        "only_manual": sample_dataframe(
            dataframe.loc[
                (dataframe["source"] == "youtube_video")
                | (dataframe["source"] == "eating_sound_collection")
            ]
        ),
    }


def train(yamnet_model, dataframe):
    class_to_id = {k: i for i, k in enumerate(dataframe["variant"].unique())}
    classes = list(class_to_id.keys())

    train_split, validate_split, _ = training.split_dataframe(dataframe)
    tf_train_split = training.preprocess_dataframe(
        yamnet_model, train_split, class_to_id
    )
    tf_validate_split = training.preprocess_dataframe(
        yamnet_model, validate_split, class_to_id
    )

    return training.train(tf_train_split, tf_validate_split, len(classes))


def visualize_metrics(class_to_id, results, title):
    y_true, y_pred = results["variant"], results["predicted"]
    y_true = y_true.map(lambda x: class_to_id[x]).values
    y_pred = y_pred.map(lambda x: class_to_id[x]).values

    # TODO: don't believe it's using the right data
    # correct = [int(a == b) for a, b in zip(y_true, y_pred)]
    # display = RocCurveDisplay.from_predictions(
    # correct, results["predicted_score"].values
    # )
    # display.plot()

    visualize.ROC_curve(class_to_id, y_true, y_pred, results, title)

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

    visualize.confusion_matrix(class_to_id, y_true, y_pred, results, title)


def dataframe_summary(dataframe: pd.DataFrame):

    df = dataframe.copy()

    df["duration_sec"] = df["path"].apply(lambda path: utils.get_audio_duration(audio_path=path))

    return {
        "value_counts": df["variant"].value_counts(),
        "average_length": df["duration_sec"].mean(),  # TODO
        "median_length": df["duration_sec"].median(),
        "stats": df["duration_sec"].describe().to_dict(),
    }


def metrics(dataframe, model_name):
    model = tf.keras.models.load_model(os.path.join(MODEL_DIR, model_name))
    model.summary()

    train_split, validate_split, test_split = training.split_dataframe(dataframe)

    # TODO: make into dataframe and output to csv
    summaries = {
        "train": dataframe_summary(train_split),
        "validate": dataframe_summary(validate_split),
        "test": dataframe_summary(test_split),
    }
    for name, summary in summaries.items():
        print(f"{name} Summary", summary)

    class_to_id = {k: i for i, k in enumerate(dataframe["variant"].unique())}
    results = result.predict(test_split, model, class_to_id)
    visualize_metrics(class_to_id, results, model_name)

    results.to_csv(os.path.join(RESULT_DIR, model_name + ".csv"))


def main(args):
    if args.train:
        if not os.path.exists(MODEL_DIR):
            os.mkdir(MODEL_DIR)

        yamnet_model = training.load_yamnet()

        dataframes = dataframe_versions()
        for name, dataframe in dataframes.items():
            model = train(yamnet_model, dataframe)
            training.save_simple(
                yamnet_model,
                model,
                os.path.join(MODEL_DIR, name),
            )

    if args.test:
        if not os.path.exists(RESULT_DIR):
            os.mkdir(RESULT_DIR)

        dataframes = dataframe_versions()
        for name, dataframe in dataframes.items():
            metrics(dataframe, name)


parser = argparse.ArgumentParser(description="UB-NSSD YAMNet transfer learning model")
parser.add_argument("--train", action="store_true", help="Whether to train the model")
parser.add_argument("--test", action="store_true", help="Whether to test the model")
args = parser.parse_args()

main(args)
