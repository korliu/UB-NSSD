import argparse

import pandas as pd
import result
import tensorflow as tf
import train
from matplotlib import pyplot as plt
from sklearn.metrics import RocCurveDisplay

DATASET_PATH = "datasets/all_data.csv"
MODEL_PATH = "nssd_model"
MODEL_WITH_INTAKE_PATH = "nssd_intake_model"
RESULT_PATH = "outputs/yamnet_analysis.csv"
RESULT_WITH_INTAKE_PATH = "outputs/yamnet_food_intake_analysis.csv"

parser = argparse.ArgumentParser(description="UB-NSSD YAMNet transfer learning model")
parser.add_argument("--train", action="store_true", help="Whether to train the model")
parser.add_argument("--test", action="store_true", help="Whether to test the model")
args = parser.parse_args()

if args.train:
    yamnet_model = train.load_yamnet()

    dataframe = pd.read_csv(DATASET_PATH)
    class_to_id = {k: i for i, k in enumerate(dataframe["variant"].unique())}
    classes = list(class_to_id.keys())

    dataframe_no_food_intake = dataframe.loc[
        dataframe["source"] != "food_intake_dataset"
    ]
    train_split, validate_split, _ = train.split_dataframe(dataframe_no_food_intake)
    tf_train_split = train.preprocess_dataframe(yamnet_model, train_split, class_to_id)
    tf_validate_split = train.preprocess_dataframe(
        yamnet_model, validate_split, class_to_id
    )

    model = train.train(tf_train_split, tf_validate_split, len(classes))
    train.save_simple(yamnet_model, model, MODEL_PATH)

    # TODO: remove boilerplate

    train_split, validate_split, _ = train.split_dataframe(dataframe)
    tf_train_split = train.preprocess_dataframe(yamnet_model, train_split, class_to_id)
    tf_validate_split = train.preprocess_dataframe(
        yamnet_model, validate_split, class_to_id
    )

    model = train.train(tf_train_split, tf_validate_split, len(classes))
    train.save_simple(yamnet_model, model, MODEL_WITH_INTAKE_PATH)


if args.test:
    dataframe = pd.read_csv(DATASET_PATH)

    # model = tf.saved_model.load(MODEL_PATH)
    model = tf.keras.models.load_model(MODEL_PATH)
    model.summary()

    # dataframe = dataframe.loc[dataframe["source"] != "food_intake_dataset"]

    test = result.predict(dataframe, model)
    y_true, y_pred = test["variant"], test["predicted"]

    class_to_id = {k: i for i, k in enumerate(dataframe["variant"].unique())}
    y_true = y_true.map(lambda x: class_to_id[x]).values
    y_pred = y_pred.map(lambda x: class_to_id[x]).values

    correct = [int(a == b) for a, b in zip(y_true, y_pred)]

    display = RocCurveDisplay.from_predictions(correct, test["predicted_score"].values)
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

    test.to_csv(RESULT_PATH)

    # TODO: show metrics
    model = tf.keras.models.load_model(MODEL_WITH_INTAKE_PATH)
    test = result.predict(dataframe, model)
    test.to_csv(RESULT_WITH_INTAKE_PATH)
