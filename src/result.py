import numpy as np
import tensorflow as tf
import train


def predict(dataframe, model):
    _, _, test = train.split_dataframe(dataframe)
    return predict_split(
        test,
        model,
        dataframe["variant"].unique(),
    )


# mutates the passed dataframe with the `predicted` and `predicted_score` columns
def predict_split(dataframe, model, classes):
    for i, row in dataframe.iterrows():
        waveform = train.load_wav_16k_mono(row["path"])
        results = model(waveform)
        top_class = tf.math.argmax(results)
        class_probabilities = tf.nn.softmax(results, axis=-1)

        dataframe.at[i, "predicted"] = classes[top_class]
        dataframe.at[i, "predicted_score"] = class_probabilities[top_class].numpy()

    return dataframe
