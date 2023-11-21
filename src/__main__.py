import argparse
import os

import imblearn
import pandas as pd
import result
import sklearn
import tensorflow as tf
import training
import utils
import visualize
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

DATASET_PATH = "datasets/all_data.csv"
MODEL_DIR = "models"
RESULT_DIR = "outputs"

BALANCE_MARGIN = 0.1  # 10%


# TODO: lots of options for over/undersampling, should check them out
def undersample_dataframe(dataframe):
    return RandomUnderSampler(
        sampling_strategy="not minority", random_state=1
    ).fit_resample(dataframe, dataframe["variant"])[0]


def oversample_dataframe(dataframe):
    return RandomOverSampler(
        sampling_strategy="not majority", random_state=1
    ).fit_resample(dataframe, dataframe["variant"])[0]


def encode_dataframe(dataframe):
    encoders = {}
    for column in dataframe.columns:
        encoder = sklearn.preprocessing.LabelEncoder()
        encoders[column] = encoder

        encoder.fit(dataframe[column])
        dataframe[column] = encoder.transform(dataframe[column])

    return encoders


def decode_dataframe(dataframe, encoders):
    for column in dataframe.columns:
        dataframe[column] = encoders[column].inverse_transform(dataframe[column])


def sample_dataframe(dataframe, algorithm):
    print(dataframe_summary(dataframe))

    encoders = encode_dataframe(dataframe)
    new = algorithm.fit_resample(dataframe, dataframe["variant"])[0]
    decode_dataframe(new, encoders)

    print(dataframe_summary(new))

    return new


# returns a list of dataframes to create models for
def dataframe_versions():
    dataframe = pd.read_csv(DATASET_PATH)
    dataframe = dataframe.loc[dataframe["variant"] != "other"]

    algorithms = {
        "RandomOverSampler,": imblearn.over_sampling.RandomOverSampler(
            sampling_strategy="auto", random_state=1
        ),
        "SMOTE,": imblearn.over_sampling.SMOTE(
            sampling_strategy="auto", random_state=1
        ),
        # "SMOTENC,": imblearn.over_sampling.SMOTENC(
        #     categorical_features="auto", sampling_strategy="auto", random_state=1
        # ),
        "SMOTEN,": imblearn.over_sampling.SMOTEN(
            sampling_strategy="auto", random_state=1
        ),
        "ADASYN,": imblearn.over_sampling.ADASYN(
            sampling_strategy="auto", random_state=1
        ),
        "BorderlineSMOTE,": imblearn.over_sampling.BorderlineSMOTE(
            sampling_strategy="auto", random_state=1
        ),
        # "KMeansSMOTE,": imblearn.over_sampling.KMeansSMOTE(
        #     sampling_strategy="auto", random_state=1
        # ),
        "SVMSMOTE,": imblearn.over_sampling.SVMSMOTE(
            sampling_strategy="auto", random_state=1
        ),
        "CondensedNearestNeighbour,": imblearn.under_sampling.CondensedNearestNeighbour(
            sampling_strategy="auto", random_state=1
        ),
        # "EditedNearestNeighbours,": imblearn.under_sampling.EditedNearestNeighbours(
        #     sampling_strategy="auto",
        # ),
        # "RepeatedEditedNearestNeighbours,": imblearn.under_sampling.RepeatedEditedNearestNeighbours(
        #     sampling_strategy="auto",
        # ),
        "AllKNN,": imblearn.under_sampling.AllKNN(
            sampling_strategy="auto",
        ),
        "InstanceHardnessThreshold,": imblearn.under_sampling.InstanceHardnessThreshold(
            sampling_strategy="auto", random_state=1
        ),
        "NearMiss,": imblearn.under_sampling.NearMiss(
            sampling_strategy="auto",
        ),
        # "NeighbourhoodCleaningRule,": imblearn.under_sampling.NeighbourhoodCleaningRule(
        #     sampling_strategy="auto",
        # ),
        "OneSidedSelection,": imblearn.under_sampling.OneSidedSelection(
            sampling_strategy="auto", random_state=1
        ),
        "RandomUnderSampler,": imblearn.under_sampling.RandomUnderSampler(
            sampling_strategy="auto", random_state=1
        ),
        "TomekLinks,": imblearn.under_sampling.TomekLinks(
            sampling_strategy="auto",
        ),
        "SMOTEENN": imblearn.combine.SMOTEENN(sampling_strategy="auto", random_state=1),
        "SMOTETomek": imblearn.combine.SMOTETomek(
            sampling_strategy="auto", random_state=1
        ),
    }

    only_intakes = {}
    for name, algorithm in algorithms.items():
        print(name)
        only_intakes["only_intake_" + name] = sample_dataframe(
            dataframe.loc[dataframe["source"] == "food_intake_dataset"], algorithm
        )

    return {
        "all": dataframe,
        # "only_intake": oversample_dataframe(
        #     dataframe.loc[dataframe["source"] == "food_intake_dataset"]
        # ),
        # "only_manual":
        #     dataframe.loc[
        #         (dataframe["source"] == "youtube_video")
        #         | (dataframe["source"] == "eating_sound_collection")
        #     ]
        # ,
        # "all_oversampled": oversample_dataframe(dataframe),
        # "only_intake_oversampled": oversample_dataframe(
        #     dataframe.loc[dataframe["source"] == "food_intake_dataset"]
        # ),
        # "only_manual_oversampled": oversample_dataframe(
        #     dataframe.loc[
        #         (dataframe["source"] == "youtube_video")
        #         | (dataframe["source"] == "eating_sound_collection")
        #     ]
        # ),
        # "all_undersampled": undersample_dataframe(dataframe),
        # "only_intake_undersampled": undersample_dataframe(
        #     dataframe.loc[dataframe["source"] == "food_intake_dataset"]
        # ),
        # "only_manual_undersampled": undersample_dataframe(
        #     dataframe.loc[
        #         (dataframe["source"] == "youtube_video")
        #         | (dataframe["source"] == "eating_sound_collection")
        #     ]
        # ),
    } | only_intakes


def train(yamnet_model, dataframe, class_to_id):
    train_split, validate_split, _ = training.split_dataframe(dataframe)
    tf_train_split = training.preprocess_dataframe(
        yamnet_model, train_split, class_to_id
    )
    tf_validate_split = training.preprocess_dataframe(
        yamnet_model, validate_split, class_to_id
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

    train_split, validate_split, test_split = training.split_dataframe(dataframe)

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


parser = argparse.ArgumentParser(description="UB-NSSD YAMNet transfer learning model")
parser.add_argument("--train", action="store_true", help="Whether to train the model")
parser.add_argument("--test", action="store_true", help="Whether to test the model")
args = parser.parse_args()

main(args)
