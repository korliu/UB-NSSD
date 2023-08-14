import whisper_at
import os
import sys
import json
import torchaudio
import numpy as np
import torch

## IDEA:
'''
1. run whisper-at on each event, see if it has the correct audio tag
    - since whisper-at does not have swallowing but does have chewing and biting, focus on chewing and biting and see what it outputs for swallowing/undefined events
    - Related audio tags for food-related stuff?:
        labels can be found by:
            -`whisper_at.print_label_name()` or 
            - `{.venv}\Lib\site-packages\whisper_at\assets\class_labels_indices.csv` file
        * 54	/m/03cczk	Chewing, mastication
        * 55	/m/07pdhp0	Biting
        56	/m/0939n_	Gargling
        57	/m/01g90h	Stomach rumble
        58	/m/03q5_w	Burping, eructation
2. (maybe) run whisper-at on audio split of each from annotated data?
3. compare with annotated_data event json files
    - compare event labels?
'''
## 

## OUTPUT CSV

# OUTPUT_DIR = os.path.join("outputs")
# if not os.path.exists(OUTPUT_DIR):
    # os.mkdir(OUTPUT_DIR)

'''
use pandas to create csv file
food, speaker, audio_id, event_id, expected_tag, detected_tags, contains_correct_tag
[food], [speaker/subject id], [event number], [expected tag, based on annotated_data], [the audio tags whisper-at outputs/detected], [True/False if the expected_tag is in detected_tags]
audio_file = "{audio_id}.wav"
annotated_data_file = "Annotated_Data_{audio_id}.json"
'''

##


# ".." not needed because we are running `python ./src/whisper_at_analysis` from main (UB-NSSD) folder
DATASET = os.path.join("datasets", "Food Intake Dataset")
DATA_SAMPLE_RATE = 8000

MODEL_TYPE = "medium"

# default model type, 1st arg is the model type
if len(sys.argv) <= 1:
    print("No model arg, using medium model as default...")

else:
    models = set(["medium", "large-v1","small.en", "base.en", "tiny.en"])
    model_arg = sys.argv[1]

    if model_arg not in models:
        raise RuntimeError(f"Incorrect model type, available model types: {models}")

    MODEL_TYPE = model_arg


# test audio to figure out best way to analyze this
test_audio = os.path.join(DATASET, "Apple", "BY", "1.wav")


wh_model = whisper_at.load_model(MODEL_TYPE)

## CONFIG for whisper-at, expect model_type which is defined by CLI or default medium model
# hard-coded for now as im not sure if we are gonna use whisper-at in the future
AUDIO_TIME_TAGGING_RES = 10
TOP_K = 5
P_THRESHOLD = -5.0
LABELS = list(range(527)) # [(indices of labels we want)] ex: [54, 55] to see only biting/chewing labels (their label ids)
##

# testing event split audio
# 1.864, 2.203; 1st event in {DATASET}/Apple/BY/1.wav
event_audio, _ = torchaudio.load(
    test_audio,
    frame_offset=int(1.864*DATA_SAMPLE_RATE),
    num_frames=int((2.203-1.864)*DATA_SAMPLE_RATE),
    )


temp_audio_path = os.path.join("src", "temp_audio.wav")
torchaudio.save(temp_audio_path,event_audio,8000,bits_per_sample=16)

audio_input = temp_audio_path

# print(f"Audio Input: {audio_input}")

transcribe_result = wh_model.transcribe(
    audio=audio_input,
    at_time_res=0.8, # most events are around this length
    word_timestamps=False,
)

parsed_results = whisper_at.parse_at_label(
    result=transcribe_result,
    language="en", # default is "follow_asr", but "en" because it outputted some korean output when i tested some audio"
    top_k=TOP_K,
    p_threshold=P_THRESHOLD,
    include_class_list=LABELS, # default, all labels, change to "[54,55]" to see only biting and chewing label
)

print(parsed_results)