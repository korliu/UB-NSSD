import tensorflow as tf
import training


def predict(dataframe, model, classes):
    class_to_id = {k: i for i, k in enumerate(dataframe["variant"].unique())}
    classes = list(class_to_id.keys())

    dataset = tf.data.Dataset.from_tensor_slices((dataframe["path"]))
    dataset = dataset.map(lambda path: training.load_wav_16k_mono(path))

    predictions = model.predict(dataset)
    chunk_size = len(classes)

    for start in range(0, len(predictions), chunk_size):
        prediction = predictions[start : (start + chunk_size)]

        top_class = tf.math.argmax(prediction)
        class_probabilities = tf.nn.softmax(prediction, axis=-1)

        i = start / chunk_size
        dataframe.at[i, "predicted"] = classes[top_class]

        for id in range(len(prediction)):
            dataframe.at[i, f"{classes[id]}_score"] = class_probabilities[id].numpy()

    return dataframe
