import tensorflow as tf

import train

def predict(dataframe, model):
    class_to_id = {k: i for i, k in enumerate(dataframe["variant"].unique())}
    classes = list(class_to_id.keys())

    _, _, test = train.split_dataframe(dataframe)
    return train.predict_split(test, model, classes, class_to_id)

# mutates the passed dataframe with the `predicted` and `predicted_score` columns
def predict_split(dataframe, model, classes, class_to_id):
    dataset = train.preprocess_dataframe(dataframe)
    for i, row in enumerate(dataset):
        # TODO: should already be processed
        # waveform = train.load_wav_16k_mono(row[0])
        results = model(row[0])
        top_class = tf.math.argmax(results)
        class_probabilities = tf.nn.softmax(results, axis=-1)

        dataframe.at[i, "predicted"] = classes[top_class]
        dataframe.at[i, "predicted_score"] = class_probabilities[top_class].numpy()

    return dataframe
