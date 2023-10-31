import os
import shutil
import csv
import numpy as np
import random
from natsort import os_sorted
import soundfile


headers = ["path","variant","food","sex","other_labels","source"]

def new_row(path,variant,food,sex,other_labels,source):

    return [path,variant,food,sex,other_labels,source]


foods = set(["aloe","burger","candied_fruits","cabbage","carrots","chocolate","fries","grapes","pickles","ribs","salmon","wings"])


DATASET = os.path.join("datasets","eating_sound_collection")
OUTPUT_DATASET = os.path.join("datasets","eating_foods_sound")

if not os.path.exists(OUTPUT_DATASET):
    os.mkdir(OUTPUT_DATASET)

ub_zip = 14260
random.seed(a=ub_zip)

OUTPUT_CSV = os.path.join(OUTPUT_DATASET,"foods_data.csv")

csv_output = open(OUTPUT_CSV,'w+',newline='')
csv_writer = csv.writer(csv_output)
csv_writer.writerow(headers)

for food_sound in os.scandir(DATASET):

    food_name = food_sound.name
    if food_name not in foods:
        continue

    food_data_path = os.path.join(DATASET,food_name)

    food_audio_clips = os.listdir(food_sound)

    random_10 = os_sorted(random.sample(food_audio_clips,k=10))

    output_dir = os.path.join(OUTPUT_DATASET,food_name)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for audio_name in random_10:

        audio_path = os.path.join(DATASET,food_name,audio_name)

        output_path = os.path.join(output_dir,audio_name)

        to_path = shutil.copy(src=audio_path,dst=output_path)

        csv_row = new_row(to_path,"______",food_name,"na","")

        csv_writer.writerow(csv_row)
