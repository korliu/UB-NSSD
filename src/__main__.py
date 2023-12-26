import argparse
import os

import pandas as pd
import preprocess
import result
import tensorflow as tf
import training
import utils
import visualize

DATASET_PATH = "datasets/all_data.csv"
MODEL_DIR = "models"
RESULT_DIR = "outputs"

BALANCE_MARGIN = 0.1  # 10%


# returns a list of dataframes to create models for
def dataframe_versions():
    dataframe = pd.read_csv(DATASET_PATH)
    dataframe = dataframe.loc[dataframe["variant"] != "other"]
    return {
        "all": dataframe,
        "only_intake": dataframe.loc[dataframe["source"] == "food_intake_dataset"],
        # "only_manual":
        #     dataframe.loc[
        #         (dataframe["source"] == "youtube_video")
        #         | (dataframe["source"] == "eating_sound_collection")
        #     ]
        # ,
    }


def train(yamnet_model, dataframe, class_to_id):
    tf_train_split, tf_validate_split, _ = training.preprocess_dataframes(
        yamnet_model, class_to_id, preprocess.split(dataframe)
    )

    return training.train(tf_train_split, tf_validate_split, len(class_to_id))


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

    metric_results = dict()

    auc = tf.keras.metrics.AUC()
    auc.update_state(y_true, y_pred)
    auc_score = auc.result().numpy()
    metric_results['auc-score'] = auc_score
    # print(f"AUC: {auc_score}")

    precision = tf.keras.metrics.Precision()
    precision.update_state(y_true, y_pred)
    precision_score = precision.result().numpy()
    metric_results['precision-score'] = precision_score
    # print(f"Precision: {precision_score}")

    recall = tf.keras.metrics.Recall()
    recall.update_state(y_true, y_pred)
    recall_score = recall.result().numpy()
    metric_results['recall-score'] = recall_score
    # print(f"Recall: {recall_score}")

    # f1 = tf.keras.metrics.F1Score()
    # f1.update_state(y_true, y_pred)
    # f1.result().numpy()
    # print(f"F1: {f1.result().numpy()}")
    # print(y_true,y_pred)
    f1_scores = visualize.get_f1_score(class_to_id, y_true, y_pred, title)

    for f1_type, f1_score in f1_scores.items():
        metric_results[f"f1-score-{f1_type}"] = f1_score

    
    visualize.plot_metrics(metric_results, title)


    visualize.confusion_matrix(class_to_id, y_true, y_pred, results, title)


    


def dataframe_summary(dataframe: pd.DataFrame):
    df = dataframe.copy()

    df["duration_sec"] = df["path"].apply(
        lambda path: utils.get_audio_duration(audio_path=path)
    )

    return {
        "value_counts": df["variant"].value_counts().to_dict(),
        "average_length": df["duration_sec"].mean(),  # TODO
        "median_length": df["duration_sec"].median(),
        # "stats": df["duration_sec"].describe().to_dict(),
    }


def metrics(dataframe, model_name, class_to_id):
    model = tf.keras.models.load_model(os.path.join(MODEL_DIR, model_name))
    model.summary()

    train_split, validate_split, test_split = preprocess.split(dataframe)

    # TODO: make into dataframe and output to csv
    summaries = {
        "train": dataframe_summary(train_split),
        "validate": dataframe_summary(validate_split),
        "test": dataframe_summary(test_split),
    }
    for name, summary in summaries.items():
        print(f"{name} Summary for {model_name}", summary)

    classes = list(class_to_id.keys())
    results = result.predict(test_split, model, classes)
    visualize_metrics(class_to_id, results, model_name)

    results.to_csv(os.path.join(RESULT_DIR, model_name + ".csv"))


def main(args):
    if args.train:
        if not os.path.exists(MODEL_DIR):
            os.mkdir(MODEL_DIR)

        yamnet_model = training.load_yamnet()

        dataframes = dataframe_versions()
        class_to_id = {
            k: i for i, k in enumerate(dataframes["all"]["variant"].unique())
        }
        for name, dataframe in dataframes.items():
            model = train(yamnet_model, dataframe, class_to_id)
            training.save_simple(
                yamnet_model,
                model,
                os.path.join(MODEL_DIR, name),
            )

    if args.test:
        if not os.path.exists(RESULT_DIR):
            os.mkdir(RESULT_DIR)

        dataframes = dataframe_versions()
        class_to_id = {
            k: i for i, k in enumerate(dataframes["all"]["variant"].unique())
        }
        for name, dataframe in dataframes.items():
            metrics(dataframe, name, class_to_id)


parser = argparse.ArgumentParser(description="Training UB-NSSD YAMNet transfer learning model")
parser.add_argument("--train", action="store_true", help="Whether to train the model")
parser.add_argument("--test", action="store_true", help="Whether to test the model")
args = parser.parse_args()

main(args)
