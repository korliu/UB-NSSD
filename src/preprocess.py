import imblearn
import pandas as pd
import sklearn

SAMPLE_SEED = 1
SHUFFLE_SEED = 1

TRAIN_RATIO = 0.6
VALIDATE_RATIO = 0.2
TEST_RATIO = 0.2


def encode(dataframe):
    encoders = {}
    for column in dataframe.columns:
        encoder = sklearn.preprocessing.LabelEncoder()
        encoders[column] = encoder

        encoder.fit(dataframe[column])
        dataframe[column] = encoder.transform(dataframe[column])

    return encoders


def decode(dataframe, encoders):
    for column in dataframe.columns:
        dataframe[column] = encoders[column].inverse_transform(dataframe[column])


def sample(dataframe):
    algorithm = imblearn.over_sampling.SMOTEN(
        sampling_strategy="auto", random_state=SAMPLE_SEED
    )
    encoders = encode(dataframe)
    new = algorithm.fit_resample(dataframe, dataframe["variant"])[0]
    decode(new, encoders)
    return new


def split_field(dataframe, field: str, splits: list[float]):
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

            subdataframes[i] = pd.concat([subdataframes[i], subset.iloc[start:end]])
            start += subset_size

    # make each dataframe start at index 1
    for i, dataframe in enumerate(subdataframes):
        subdataframes[i] = dataframe.reset_index()

    return subdataframes


def split(dataframe):
    dataframe = dataframe.sample(frac=1, random_state=SHUFFLE_SEED)

    splits = split_field(
        dataframe, "variant", [TRAIN_RATIO, VALIDATE_RATIO, TEST_RATIO]
    )
    for i, split in enumerate(splits):
        splits[i] = sample(split)

    return splits
