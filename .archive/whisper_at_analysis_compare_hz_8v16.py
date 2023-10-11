import whisper_at
import os
import sys
import json
import torchaudio
import numpy as np
import pandas as pd
import torch
import random

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
        499 Crunch
2. (maybe) run whisper-at on audio split of each from annotated data?
3. compare with annotated_data event json files
    - compare event labels?
'''
## 

## OUTPUT CSV

OUTPUT_DIR = os.path.join("outputs")
if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)
OUTPUT_FILENAME = "whisper_at_analysis_compare_8kv16khz.csv"
OUTPUT = os.path.join(OUTPUT_DIR,OUTPUT_FILENAME)

if os.path.exists(OUTPUT):
    raise RuntimeError(f"Output already exists, exited to not overwrite data. Data is randomized each run.")

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
    print(f"No model arg, using {MODEL_TYPE} model as default...")

else:
    models = set(["medium", "large-v1","small.en", "base.en", "tiny.en"])
    model_arg = sys.argv[1]


    if model_arg not in models:
        raise RuntimeError(f"Incorrect model type, available model types: {models}")
    
    print(f"Using \'{model_arg}\' model for whisper...")

    MODEL_TYPE = model_arg


# test audio to figure out best way to analyze this
test_audio = os.path.join(DATASET, "Apple", "BY", "1.wav")

wh_model = whisper_at.load_model(MODEL_TYPE)

## CONFIG for whisper-at, expect model_type which is defined by CLI or default medium model
# hard-coded for now as im not sure if we are gonna use whisper-at in the future
AUDIO_TIME_TAGGING_RES = 0.8 # most events are around this length
TOP_K = 3
P_THRESHOLD = -5.0 # the lower, the more sensitive the audio tagging is
PRIO_LABELS = [54,55,499] # [(indices of labels we want)] ex: [54, 55] to see only biting/chewing labels (their label ids)
ALL_LABELS = list(range(527))
##


labels = ['food', 'speaker', 'audio_path', 'event_num', 
          'start_time_sec', 'end_time_sec', 'duration', 
          'expected_tag', 
          'detected_tags_8k', 'detected_tags_16k',
          'contains_correct_tag_8k', 'contains_correct_tag_16k']

data_dict = {label: list() for label in labels}

def add_to_datarow(
        food_name: str, speaker_name: str, audio_id: str, event_num: str, 
        start_time_sec: float, end_time_sec: float, duration: float,
        expected_tag: str, 
        detected_tags_8k: list, detected_tags_16k, 
        contains_correct_tag_8k: bool, contains_correct_tag_16k: bool) -> None:
    
    data_dict['food'].append(food_name)
    data_dict['speaker'].append(speaker_name)
    data_dict['audio_path'].append(f"{audio_id}.wav")
    data_dict['event_num'].append(event_num)
    data_dict['start_time_sec'].append(start_time_sec)
    data_dict['end_time_sec'].append(end_time_sec)
    data_dict['duration'].append(duration)
    data_dict['expected_tag'].append(expected_tag)
    data_dict['detected_tags_8k'].append(detected_tags_8k)
    data_dict['detected_tags_16k'].append(detected_tags_16k)
    data_dict['contains_correct_tag_8k'].append(contains_correct_tag_8k)
    data_dict['contains_correct_tag_16k'].append(contains_correct_tag_16k)



# loop over {dataset}\{food_folder}\{speaker_samples}\{data_file}

LIMIT = float('inf') # limit on number of files

files_scanned = 0
for food_folder in os.scandir(DATASET):
    food_name, _ = food_folder.name, food_folder.path
    if not food_folder.is_dir:
        continue
    for speaker_samples in os.scandir(food_folder):
        speaker_name, speaker_folder = speaker_samples.name, speaker_samples.path
        if not speaker_samples.is_dir:
            continue
        for data_file in os.scandir(speaker_samples):
            file_name, file_ext = os.path.splitext(data_file.name)
            
            # skips non ".wav" files
            if not data_file.is_file() or file_ext != ".wav":
                continue

            audio_id = file_name
            audio_data = f"Annotated_Data_{audio_id}.json"

            # some audio files dont have annotated data, skip those
            input_audio = os.path.join(speaker_folder,f"{audio_id}.wav")
            annotated_data = os.path.join(speaker_folder,audio_data)
            
            if not os.path.exists(annotated_data):
                continue
            
            with open(annotated_data,'r',encoding='utf-8') as f:
                json_data = json.load(f)

                audio_sr, audio_events, event_headers = json_data['audio_sampling_frequency'], json_data['events'], json_data['events_headers']

                e2l, l2e = json_data['event2label'], json_data['label2event']


                N_RANDOM = 2
                random.shuffle(audio_events)
                audio_events = sorted(audio_events[:N_RANDOM], key=lambda x:x[0])
                # print(audio_events)
                
                for e in audio_events:
                    
                    if files_scanned >= LIMIT:
                        break
                    
                    # event = dict(zip(event_headers,e)) 
                    event_num, event_label, start_time_sec, end_time_sec = e

                    audio_sr = int(audio_sr)
                    start_time_sec = float(start_time_sec)
                    end_time_sec = float(end_time_sec)
                    duration_sec = end_time_sec-start_time_sec

                    # splice the audio to be only the event
                    event_audio, _ = torchaudio.load(
                        filepath=input_audio,
                        frame_offset=int(start_time_sec*audio_sr),
                        num_frames=int(duration_sec*audio_sr),
                    )

                    # save audio to temporary file, unsure how to pass `event_audio`` directly into whisper (ran into type issues), so just let whisper_at handle it by itself by providing path to audio file
                    temp_audio_path_8k = os.path.join("src","temp_audio_8k.wav")
                    temp_audio_path_16k = os.path.join("src", "temp_audio_16k.wav")

                    # save as 16k hz, 16khz version is just sped up x2
                    torchaudio.save(temp_audio_path_8k,event_audio,sample_rate=8000)
                    torchaudio.save(temp_audio_path_16k,event_audio,sample_rate=16000)

                    ########## 8k ##############
                    whisper_transcription_result_8k = wh_model.transcribe(
                        audio=temp_audio_path_8k,
                        at_time_res=AUDIO_TIME_TAGGING_RES,
                        initial_prompt="", # can prompt engineer with NSS if we want to
                        word_timestamps=False,
                    )

                    # transcribe_result.keys():
                        # 'text', 'segments', 'language', 'at_time_res', 'audio_tag'
                    # result_text, text_segments = whisper_transcription_results['text'], whisper_transcription_results['segments']

                    parsed_results_8k = whisper_at.parse_at_label(
                        result=whisper_transcription_result_8k,
                        language='en', # default is "follow_asr", but "en" because it outputted some korean output when i tested some audio"
                        top_k=TOP_K,
                        p_threshold=P_THRESHOLD,
                        include_class_list=PRIO_LABELS,
                    )

                    # collect all tags
                    correct_tag_8k = True
                    flatten_tags_8k = [tag for segment in parsed_results_8k for tag in segment['audio tags']]
                    
                    # no inital tags are empty, see what whisper-at tags them as
                    # reparse with all labels included
                    if not flatten_tags_8k:
                        correct_tag_8k = False
                        parsed_results_8k = whisper_at.parse_at_label(
                            result=whisper_transcription_result_8k,
                            language='en', # default is "follow_asr", but "en" because it outputted some korean output when i tested some audio"
                            top_k=TOP_K,
                            p_threshold=P_THRESHOLD,
                            include_class_list=ALL_LABELS,
                        )
                        flatten_tags_8k = [tag for segment in parsed_results_8k for tag in segment['audio tags']]

                    # tag = ([tag], [logit val])
                    # if there were multiple timeframes, sort by probability/logit value 
                    flatten_tags_8k = sorted(flatten_tags_8k,key=lambda x: x[1], reverse=True)
                    # grab tags only and remove dupes
                    flatten_tags_8k = list(set([t[0] for t in flatten_tags_8k]))

                    ############# 16k #################
                    whisper_transcription_result_16k = wh_model.transcribe(
                        audio=temp_audio_path_16k,
                        at_time_res=AUDIO_TIME_TAGGING_RES,
                        initial_prompt="", # can prompt engineer with NSS if we want to
                        word_timestamps=False,
                    )

                    # transcribe_result.keys():
                        # 'text', 'segments', 'language', 'at_time_res', 'audio_tag'
                    # result_text, text_segments = whisper_transcription_results['text'], whisper_transcription_results['segments']

                    parsed_results_16k = whisper_at.parse_at_label(
                        result=whisper_transcription_result_16k,
                        language='en', # default is "follow_asr", but "en" because it outputted some korean output when i tested some audio"
                        top_k=TOP_K,
                        p_threshold=P_THRESHOLD,
                        include_class_list=PRIO_LABELS,
                    )

                    # collect all tags
                    correct_tag_16k = True
                    flatten_tags_16k = [tag for segment in parsed_results_16k for tag in segment['audio tags']]
                    
                    # no inital tags are empty, see what whisper-at tags them as
                    # reparse with all labels included
                    if not flatten_tags_16k:
                        correct_tag_16k = False
                        parsed_results_16k = whisper_at.parse_at_label(
                            result=whisper_transcription_result_16k,
                            language='en', # default is "follow_asr", but "en" because it outputted some korean output when i tested some audio"
                            top_k=TOP_K,
                            p_threshold=P_THRESHOLD,
                            include_class_list=ALL_LABELS,
                        )
                        flatten_tags_16k = [tag for segment in parsed_results_16k for tag in segment['audio tags']]

                    # tag = ([tag], [logit val])
                    # if there were multiple timeframes, sort by probability/logit value 
                    flatten_tags_16k = sorted(flatten_tags_16k,key=lambda x: x[1], reverse=True)
                    # grab tags only and remove dupes
                    flatten_tags_16k = list(set([t[0] for t in flatten_tags_16k]))

                    ############

                    
                    print(f"8K Hz, Tags for \"{annotated_data}\", Event #{event_num}: {flatten_tags_8k}")
                    print(f"16K Hz, Tags for \"{annotated_data}\", Event #{event_num}: {flatten_tags_16k}")

                    add_to_datarow(
                        food_name=food_name,speaker_name=speaker_name, audio_id=audio_id, event_num=event_num,
                        start_time_sec=start_time_sec, end_time_sec=end_time_sec, duration=duration_sec,
                        expected_tag=l2e[str(event_label)],
                        detected_tags_8k=flatten_tags_8k, detected_tags_16k=flatten_tags_16k,
                        contains_correct_tag_8k=correct_tag_8k, contains_correct_tag_16k=correct_tag_16k,
                    )
                    files_scanned += 1


# remove the temp audio file
if os.path.exists(os.path.join("src","temp_audio_8k.wav")):
    os.remove(path=os.path.join("src","temp_audio_8k.wav"))

if os.path.exists(os.path.join("src","temp_audio_16k.wav")):
    os.remove(path=os.path.join("src","temp_audio_16k.wav"))


data_df = pd.DataFrame(data=data_dict)

data_df.to_csv(OUTPUT,
               index=False, # removes index column
               lineterminator="\n", 
               na_rep="N/A",
               )
