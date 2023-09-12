import infer
import utils
import whisper_at as whisper

# TODO: make path more foolproof
utils.get_audio_from_yt(
    "https://www.youtube.com/watch?v=LRvwtlkV-IQ", "datasets/yt.wav"
)

model = whisper.load_model("base.en")

transcription = model.transcribe(
    "datasets/yt.wav",
    at_time_res=0.4,  # min time res
    # word_timestamps=True, # TODO: error
)
tags = whisper.parse_at_label(
    transcription,
    language="follow_asr",
    top_k=5,
    p_threshold=-1,
    include_class_list=list(range(527)),
)

print(infer.combine(transcription, tags))
