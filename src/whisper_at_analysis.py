import whisper_at
import os
import sys
import json


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

# ".." because we are in src folder
DATASET = os.path.join("..","datasets", "Food Intake Dataset")

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
test_audio = os.path.join(DATASET, "Apple", "Apple", "BY", "1.wav")


wh_model = whisper_at.load_model(MODEL_TYPE)

## CONFIG for whisper-at, expect model_type which is defined by CLI or default medium model
AUDIO_TIME_TAGGING_RES = 10
TOP_K = 5
P_THRESHOLD = -5.0
##


