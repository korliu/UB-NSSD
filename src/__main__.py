import argparse
import csv

import pandas as pd
import tensorflow as tf

import train
import result

DATASET_PATH = "datasets/all_data.csv"
MODEL_PATH = "nssd_model"
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

    train_split, validate_split, _ = train.split_dataset(dataframe)
    tf_train_split = train.preprocess_dataframe(yamnet_model, train_split, class_to_id)
    tf_validate_split = train.preprocess_dataframe(yamnet_model, validate_split, class_to_id)

    model = train.train(tf_train_split, tf_validate_split, len(classes))
    train.save_simple(yamnet_model, model, MODEL_PATH)


if args.test:
    dataframe = pd.read_csv(DATASET_PATH)

    # model = tf.saved_model.load(MODEL_PATH)
    model = tf.keras.models.load_model(MODEL_PATH)
    model.summary()

    test = result.predict(dataframe.loc[dataframe["source"] != "food_intake"])
    y_true, y_pred = test["predicted"], test["predicted_score"]

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

    # TODO: print metrics for food intake?
    test_with_intake = result.predict(dataframe)
    test_with_intake.to_csv(RESULT_WITH_INTAKE_PATH)
