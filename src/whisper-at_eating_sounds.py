import whisper_at
import os
import sys
import json
import torchaudio
import numpy as np
import pandas as pd
import librosa


## OUTPUT CSV

OUTPUT_DIR = os.path.join("outputs")
if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)
OUTPUT_FILENAME = "whisper_at_analysis_burger_audio.csv"
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

EATING_TAGS = set(['Crunch','Biting', "Chewing, mastication"])


# ".." not needed because we are running `python ./src/whisper_at_analysis` from main (UB-NSSD) folder
DATASET = os.path.join("datasets", "archive", "clips_rd")

MODEL_TYPE = "medium"   

# default model type, 1st arg is the model type
if len(sys.argv) <= 1:
    print(f"No model arg, using \'{MODEL_TYPE}\' model as default...")

else:
    models = set(["medium", "large-v1","small.en", "base.en", "tiny.en"])
    model_arg = sys.argv[1]


    if model_arg not in models:
        raise RuntimeError(f"Incorrect model type, available model types: {models}")
    
    print(f"Using \'{model_arg}\' model for whisper...")

    MODEL_TYPE = model_arg


# test audio to figure out best way to analyze this
test_audio = os.path.join(DATASET,"aloe","aloe_1_01.wav")

wh_model = whisper_at.load_model(MODEL_TYPE)

## CONFIG for whisper-at, expect model_type which is defined by CLI or default medium model
# hard-coded for now as im not sure if we are gonna use whisper-at in the future
AUDIO_TIME_TAGGING_RES = 0.4 * 2 # 0.8 because prev data was analyzed on 0.8 too
TOP_K = 3
P_THRESHOLD = -2.0 # the lower, the more sensitive the audio tagging is
PRIO_LABELS = [54,55, 499] # [(indices of labels we want)]
# 54, 55 is chewing, biting
# 499 is crunch

ALL_LABELS = list(range(527))
##


# labels = ['food', 'speaker', 'audio_path', 'event_num', 
#           'start_time_sec', 'end_time_sec', 'duration', 
#           'expected_tag', 'detected_tags', 'contains_prio_tag']

labels = ["food", "audio_path","event_num", "start_time_sec", "end_time_sec", "duration","expected_tag","detected_tags", "contains_prio_tag"]

data_dict = {label: list() for label in labels}

def add_to_datarow(
        food_name: str = None, speaker_name: str = None, audio_id: str = None, event_num: str = None, 
        start_time_sec: float = None, end_time_sec: float = None, duration: float = None,
        expected_tag: str = None, detected_tags: list = None, contains_prio_tag: bool = None) -> None:
    
    data_dict['food'].append(food_name)
    # data_dict['speaker'].append(speaker_name)
    data_dict['audio_path'].append(f"{audio_id}.wav")
    data_dict['event_num'].append(f"{event_num}")
    data_dict['start_time_sec'].append(start_time_sec)
    data_dict['end_time_sec'].append(end_time_sec)
    data_dict['duration'].append(duration)
    data_dict['expected_tag'].append(expected_tag)
    data_dict['detected_tags'].append(detected_tags)
    data_dict['contains_prio_tag'].append(contains_prio_tag)


# loop over {dataset}\{food_folder}\{speaker_samples}\{data_file}

LIMIT = 30 # limit on number of files

files_scanned = 0

chosen_foods = set(["burger"])
for food_folder in os.scandir(DATASET):
    food_name, _ = food_folder.name, food_folder.path
    if not food_folder.is_dir or food_name not in chosen_foods:
        continue

    for data_file in os.scandir(food_folder):
        file_name, file_ext = os.path.splitext(data_file.name)
        
        # skips non ".wav" files
        if not data_file.is_file() or file_ext != ".wav":
            continue

        audio_id = file_name

        # some audio files dont have annotated data, skip those
        input_audio = os.path.join(food_folder,f"{audio_id}.wav")
        

        if files_scanned >= LIMIT:
            break


        duration_sec = librosa.get_duration(path=input_audio)

    

        whisper_transcription_result = wh_model.transcribe(
            audio=input_audio,
            at_time_res=AUDIO_TIME_TAGGING_RES,
            initial_prompt="", # can prompt engineer with NSS if we want to
            word_timestamps=False,
        )

        # whisper_transcription_result.keys():
            # 'text', 'segments', 'language', 'at_time_res', 'audio_tag'
        # result_text, text_segments = whisper_transcription_results['text'], whisper_transcription_results['segments']

        parsed_results = whisper_at.parse_at_label(
            result=whisper_transcription_result,
            language='en', # default is "follow_asr", but "en" because it outputted some korean output when i tested some audio"
            top_k=TOP_K,
            p_threshold=P_THRESHOLD,
            include_class_list=ALL_LABELS,
        )

        print(f"Results on \'{input_audio}\': ")

        print(f"\t Raw Output: {parsed_results}")

        # collect all tags
        audio_sr = librosa.get_samplerate(path=input_audio)
        print(f"audio_sr: {audio_sr}")
        for i, r in enumerate(parsed_results):
            
            time, tags = r['time'], r['audio tags']

            start, end = float(time['start']), float(time['end'])
            duration_sec = end-start


            food_audio_dir = os.path.join(OUTPUT_DIR,f"{food_name}_audio")
            if not os.path.exists(food_audio_dir):
                os.mkdir(food_audio_dir)

            event_audio, _ = torchaudio.load(
                        filepath=input_audio,
                        frame_offset=int(start*audio_sr),
                        num_frames=int(duration_sec*audio_sr),
                    )

            # save audio to temporary file, unsure how to pass `event_audio`` directly into whisper (ran into type issues), so just let whisper_at handle it by itself by providing path to audio file
            food_audio_seg_path = os.path.join(food_audio_dir,f"{audio_id}_{i}.wav")
            torchaudio.save(food_audio_seg_path,event_audio,audio_sr)


            audio_tags = tags
            # grab tags only and remove dupes
            
            print(f"\t Tags for \"{input_audio}_{i}\", {audio_tags}")

            prio_tag = False
            for t in audio_tags:
                tag, logit = t
                if tag in EATING_TAGS:
                    prio_tag = True


            add_to_datarow(
                food_name = food_name, audio_id = audio_id, event_num = i,
                start_time_sec = start, end_time_sec = end, duration = duration_sec,
                detected_tags=audio_tags, contains_prio_tag=prio_tag
            )

            files_scanned += 1
            
            if files_scanned >= LIMIT:

                data_df = pd.DataFrame(data=data_dict)
                # data_df.index.name = ""

                data_df.to_csv(OUTPUT,
                            index=False, # removes index column
                            lineterminator="\n", 
                            na_rep="N/A",
                            )
                
                exit()



data_df = pd.DataFrame(data=data_dict)
# data_df.index.name = ""

data_df.to_csv(OUTPUT,
               index=False, # removes index column
               lineterminator="\n", 
               na_rep="N/A",
               )
