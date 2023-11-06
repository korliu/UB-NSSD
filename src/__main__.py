import argparse
import os

import pandas as pd
import result
import sklearn.metrics as sk_metrics
import tensorflow as tf
import training
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

DATASET_PATH = "datasets/all_data.csv"
MODEL_DIR = "models"
RESULT_DIR = "outputs"


# returns a list of dataframes to create models for
def dataframe_versions():
    dataframe = pd.read_csv(DATASET_PATH)
    return {
        "all": dataframe,
        "no_intake": dataframe.loc[dataframe["source"] != "food_intake_dataset"],
    }


def train(yamnet_model, dataframe):
    class_to_id = {k: i for i, k in enumerate(dataframe["variant"].unique())}
    classes = list(class_to_id.keys())

    dataframe_no_food_intake = dataframe.loc[
        dataframe["source"] != "food_intake_dataset"
    ]
    train_split, validate_split, _ = training.split_dataframe(dataframe_no_food_intake)
    tf_train_split = training.preprocess_dataframe(
        yamnet_model, train_split, class_to_id
    )
    tf_validate_split = training.preprocess_dataframe(
        yamnet_model, validate_split, class_to_id
    )

    return training.train(tf_train_split, tf_validate_split, len(classes))


def visualize_metrics(class_to_id, results):
    y_true, y_pred = results["variant"], results["predicted"]
    y_true = y_true.map(lambda x: class_to_id[x]).values
    y_pred = y_pred.map(lambda x: class_to_id[x]).values

    correct = [int(a == b) for a, b in zip(y_true, y_pred)]
    display = RocCurveDisplay.from_predictions(
        correct, results["predicted_score"].values
    )
    display.plot()
    plt.show()

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

    display = ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    display.plot()
    plt.show()

    confusion_matrices = sk_metrics.multilabel_confusion_matrix(y_true, y_pred)
    for i in zip(class_to_id.keys(), confusion_matrices):
        print(f"{i[0]} -> Confusion Matrix: {i[1]}")


def metrics(dataframe, model_name):
    model = tf.keras.models.load_model(os.path.join(MODEL_DIR, model_name))
    model.summary()

    results = result.predict(dataframe, model)
    class_to_id = {k: i for i, k in enumerate(dataframe["variant"].unique())}

    visualize_metrics(class_to_id, results)
    results.to_csv(os.path.join(RESULT_DIR, model_name))


def main(args):
    if args.train:
        if not os.path.exists(MODEL_DIR):
            os.mkdir(MODEL_DIR)

        yamnet_model = training.load_yamnet()

        dataframes = dataframe_versions(
            yamnet_model,
        )
        for name, dataframe in dataframes.items():
            model = train(yamnet_model, dataframe)
            model.save_simple(
                yamnet_model,
                model,
                os.path.join(MODEL_DIR, name),
            )

    if args.test:
        if not os.path.exists(RESULT_DIR):
            os.mkdir(RESULT_DIR)

        dataframes = dataframe_versions(
            yamnet_model,
        )
        for name, dataframe in dataframes.items():
            metrics(dataframe, name)


parser = argparse.ArgumentParser(description="UB-NSSD YAMNet transfer learning model")
parser.add_argument("--train", action="store_true", help="Whether to train the model")
parser.add_argument("--test", action="store_true", help="Whether to test the model")
args = parser.parse_args()

main(args)
