import os
import pandas as pd
import json
import requests
import csv

AUDIOSET_FOLDER = os.path.join("datasets","google_audioset")
AUDIO_SET_LINKS_JSON = os.path.join(AUDIOSET_FOLDER,"google_audioset.json")

OUTPUT_FOLDER = os.path.join(AUDIOSET_FOLDER,"audioset")
if not os.path.exists(OUTPUT_FOLDER):
        os.mkdir(OUTPUT_FOLDER)


links_json_file = None
with open(AUDIO_SET_LINKS_JSON, 'r') as f:
    links_json_file = json.load(f)


labels_csv_link = links_json_file['class_labels_csv']

dataset_links = links_json_file['dataset_csv_download_links']

labels_wanted = set([54,55]) # chewing and biting, respectively


# get csv from link
def download_labels_csv(link: str, save_path: str="audioset_labels.csv") -> None:
    labels_request = requests.get(link)

    if labels_request.status_code != 200:
        raise Exception(f"Download request from {link} failed...")
    
    # save to output
    with open(os.path.join(OUTPUT_FOLDER,"audioset_labels.csv"), 'w+') as f:
        f.write(labels_request.text)

    return


def get_label2name_dict(labels_csv: str ) -> dict:
    
    id2name = dict()
    id2mid = dict()
    mid2id = dict()
    name2id = dict()


    with open(labels_csv,'r') as f:
        csv_reader = csv.reader(f)

        header = next(csv_reader)

        for data in csv_reader:
            id, mid, label_name = data

            id2name[id] = label_name

            id2mid[id] = mid
            mid2id[mid] = id

            name2id[label_name] = id

    
    return (id2name, id2mid, mid2id, name2id)
        
if not os.path.exists(os.path.join(OUTPUT_FOLDER,"audioset_labels.csv")):
    _ = download_labels_csv(labels_csv_link)
    

id2name, id2mid, mid2id, name2id = get_label2name_dict(os.path.join(OUTPUT_FOLDER,"audioset_labels.csv"))

# test_link = dataset_links['evaluation']
# test = requests.get(test_link)
# print(test.text)

OUTPUT_DATA_FOLDER = os.path.join(OUTPUT_FOLDER,"audioset_data")
if not os.path.exists(OUTPUT_DATA_FOLDER):
    os.mkdir(OUTPUT_DATA_FOLDER)

id_labels_filter = set(["54","55"])
# ids 54 and 55 are chewing and biting

print(dataset_links)
for k,v in dataset_links.items():
    dataset_name, dataset_link = k,v
    dataset_request = requests.get(dataset_link)

    if dataset_request.status_code != 200:
        print(f"Download request from {dataset_link} failed, moving on..")
        continue
    
    with open(os.path.join(OUTPUT_DATA_FOLDER,f"{dataset_name}.csv"), 'w+',newline='') as f:


        csv_writer = csv.writer(f)


        csv_reader = csv.reader(dataset_request.text.splitlines(),skipinitialspace=True)

        metadata_text = next(csv_reader) # unimportant information
        metadata_text = next(csv_reader)

        header = next(csv_reader)
        header[0] = "YTID"
        csv_writer.writerow(header)


        for row in csv_reader:
            # print(rf'{row}')
            yt_id, start_sec, end_sec, _ = row

            positive_mid_labels = row[3:][0].split(',')

            eating_video = False

            for mid_label in positive_mid_labels:
                if mid2id[mid_label] in id_labels_filter:
                    eating_video=True
                    break

            if eating_video:
                csv_writer.writerow(row)
            
            





