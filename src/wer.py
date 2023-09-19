import torch
import json
from torcheval.metrics import WordErrorRate
import infer
import whisper_at as whisper
import utils


def compute_wer(expected_text: str, predicted_text: str):
    metric = WordErrorRate()
    metric.update(expected_text, predicted_text)
    return metric.compute()


annotated = json.load(open("datasets/manually_annotated.json"))
for i, a in enumerate(annotated):
    model = whisper.load_model("base.en")

    start, end = a['start_second'], a['end_second']

    audio_path = f"datasets/{i}.wav"
    utils.get_audio_from_yt(a["yt_link"], audio_path, start, end)

    transcription = model.transcribe(
        audio_path,
        at_time_res=0.4,  # min time res
        word_timestamps=True,
    )
    tags = whisper.parse_at_label(
        transcription,
        language="en",
        include_class_list=list(range(1, 527)),
    )

    combo = infer.combine(transcription, tags)

    strings = []
    for x in combo:
        if "tag" in x:
            strings.append("<" + x["tag"].lower() + ">")
        else:
            strings.append(x["word"])

    expected = a["transcription"]
    predicted = " ".join(strings)
    wer = compute_wer(expected, predicted)

    print(f"expected: {expected}")
    print(f"predicted: {predicted}")
    print(f"{i}.wav", wer)
    print()
