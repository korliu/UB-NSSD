import whisper_at as whisper

audio_tagging_time_resolution = 10
model = whisper.load_model("large-v1")
result = model.transcribe(
    "datasets/test.wav", at_time_res=audio_tagging_time_resolution
)
# ASR Results
print(result["text"])
# Audio Tagging Results
audio_tag_result = whisper.parse_at_label(
    result,
    language="follow_asr",
    top_k=5,
    p_threshold=-1,
    include_class_list=list(range(527)),
)
print(audio_tag_result)
