import utils
import pandas as pd
import os


SOURCE_DATA = os.path.join(
    "datasets", "google_audioset", "audioset", "audioset_data", "raw_data.csv"
)
UNBALANCED = os.path.join(SOURCE_DATA, "unbalanced_train.csv")
BALANCED = os.path.join(SOURCE_DATA, "balanced_train.csv")
EVAL = os.path.join(SOURCE_DATA, "evaluation.csv")

DATASET = os.path.join("datasets", "train_yamnet")
if not os.path.exists(DATASET):
    os.mkdir(DATASET)

raw_data = pd.read_csv(SOURCE_DATA)
# print(raw_data.head())

eating_classes = ["chewing", "biting", "swallow", "other"]
class2id = {k: i for i, k in enumerate(eating_classes)}

# print(class2id)

class_id = raw_data["positive_labels"].apply(lambda name: class2id.get(name, -1))
filtered_pd = raw_data.assign(target=class_id)


# print(filtered_pd[["YTID", "start_seconds", "end_seconds"]].head())


def get_yt_audio_path(yt_id, start_sec, end_sec):
    get_yt_link = utils.get_yt_url(yt_id)

    yt_audio_data = utils.get_audio_from_yt(
        youtube_link=get_yt_link,
        save_path=os.path.join(DATASET, f"{yt_id}_{int(start_sec)}-{int(end_sec)}.wav"),
        start_second=start_sec,
        end_second=end_sec,
    )

    yt_audio_path = yt_audio_data["audio_path"]

    return yt_audio_path


yt_data_cols = zip(
    filtered_pd["YTID"], filtered_pd["start_seconds"], filtered_pd["end_seconds"]
)
yt_audio_paths = []
for ytid, start, end in yt_data_cols:
    get_path = get_yt_audio_path(ytid, start, end)

    if get_path is not None:
        yt_audio_paths.append(get_path)
