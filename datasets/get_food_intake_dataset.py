import os
import shutil
import csv
import numpy as np
import soundfile as sf
import json
from pathlib import Path

headers = ["path","variant","food","sex","other_labels","source"]

def new_row(path,variant,food,sex,other_labels,source):

    return [path,variant,food,sex,other_labels,source]

def rename_label(label: str):

    match label:

        case "bite":
            return "biting"
        case "chew":
            return "chewing"
        case "undefined":
            return "other"

        case "swallow":
            return "swallow"
        
        case _:
            raise Exception("Not a label")
    


# DATASET = os.path.join("datasets","Food Intake Dataset")
DATASET = Path("datasets", "Food Intake Dataset")
OUTPUT_DATASET = Path("datasets","food_intake_dataset")

OUTPUT_CSV = Path("datasets","food_intake_data.csv")

OUTPUT_DATASET.mkdir(exist_ok=True)

csv_output = open(OUTPUT_CSV,'w+',newline='')
csv_writer = csv.writer(csv_output)
csv_writer.writerow(headers)

for food_folder in DATASET.iterdir():


    food_name = food_folder.name

    for speaker_folder in food_folder.iterdir():

        speaker_name = speaker_folder.name

        for data_file in speaker_folder.iterdir():
            
            file_ext = data_file.suffix
            
            #skip .wav
            if file_ext != ".wav":
                continue

            audio_file = data_file
            audio_id = audio_file.stem
            audio_json_data = Path(speaker_folder,f"Annotated_Data_{audio_id}.json")

            if not audio_json_data.exists():
                continue

            with open(audio_json_data,"r") as json_f:

                audio_json = json.load(json_f)

                speaker_info = audio_json["speaker_info"]
                speaker_id, speaker_gender = speaker_info["id"], speaker_info["gender"]

                food_type = audio_json["food_type"]
                audio_sr = int(audio_json["audio_sampling_frequency"])


                id2event = audio_json["label2event"]

                # 'events_headers': ['event_number', 'event_class', 'start_time_sec', 'end_time_sec']
                audio_events = audio_json["events"]

                for e in audio_events:

                    event_number, event_class_id, start_time_sec, end_time_sec = e

                    if end_time_sec <= start_time_sec:

                        # see where those are
                        # if start_time_sec - end_time_sec >= 1.5:
                        #     print(f"Event {event_number} in {audio_json_data} has an end time before a start time")
                        #     print(e)
                        continue

                    start_frame = int(start_time_sec * audio_sr)
                    end_frame = int(end_time_sec * audio_sr)


                    audio_data, sr = sf.read(audio_file,start=start_frame, stop=end_frame)

                    food_name_data = food_type.lower().replace(" ","_")

                    data_folder = Path(OUTPUT_DATASET,food_name_data, speaker_id.lower())
                    data_folder.mkdir(parents=True,exist_ok=True)

                    data_path = f"{audio_id}_{event_number}.wav"

                    output_path = Path(data_folder, data_path).as_posix()
                    # if output_path.exists():
                    #     print(f"{output_path} EXISTS ALREADY!")

                    # print(output_path)
                    sf.write(output_path,audio_data,samplerate=audio_sr,subtype="PCM_16")

                    variant_label = rename_label(id2event[str(event_class_id)])
                    # print(variant_label)

                    csv_row = new_row(
                        path=output_path,
                        variant=variant_label,
                        food=food_name_data,
                        sex=speaker_gender.lower(),
                        other_labels="",
                        source="food_intake_dataset"
                        )
                    
                    # print(csv_row)
                    csv_writer.writerow(csv_row)