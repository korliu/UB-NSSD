import os
import pandas as pd
import numpy as np
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


def get_label2name_dict(labels_csv: str) -> dict:
    
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

eating_id_labels = {"54": "chewing", "55": "biting"}
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

        dataset = dataset_request.text.splitlines()
        # rough amount of rows (not exactly n_data, closer to n_data-3) done to collect a rough amount of data with an other tag.
        n_data = len(dataset)
        max_num_other_label = n_data//1000
        n_other = 0
        other_data = []
        n_eating = 0

        csv_reader = csv.reader(dataset,skipinitialspace=True)

        metadata_text = next(csv_reader) # unimportant information
        metadata_text = next(csv_reader)

        header = next(csv_reader)
        header[0] = "YTID"
        csv_writer.writerow(header)

        for row in csv_reader:
            # print(rf'{row}')
            yt_id, start_sec, end_sec, _ = row

            positive_mid_labels = row[3:][0].split(',')
            for i in range(len(positive_mid_labels)):
                mid_label = positive_mid_labels[i]

                positive_mid_labels[i] 

            named_labels = list(map(lambda x: id2name[mid2id[x]], positive_mid_labels))
            is_eating_video = False

            data_row = [yt_id,start_sec,end_sec]
            for i, label in enumerate(named_labels):
                if name2id[label] in eating_id_labels:
                    is_eating_video=True
                    n_eating += 1
                    data_row.append(eating_id_labels[name2id[label]])
                    break

            
            if is_eating_video:
                csv_writer.writerow(data_row)

            elif n_other <= max_num_other_label:
                n_other += 1
                data_row.append("other")
                other_data.append(data_row)

        # set seed to reproduce randomizable results
        seed = np.random.seed(123)
        np.random.shuffle(other_data)

        # roughly 10% of data will be other labels
    
        for other in other_data[:n_eating//10]:
            csv_writer.writerow(other)
        
            
            


# add manually_annotated
MANUAL_TRAIN_DATA = os.path.join("datasets","manual_train_data.csv")
MANUAL_EVAL_DATA = os.path.join("datasets","manual_eval_data.csv")



TRAIN_DATA = os.path.join(OUTPUT_DATA_FOLDER,"unbalanced_train.csv")
EVAL_DATA = os.path.join(OUTPUT_DATA_FOLDER,"evaluation.csv")


datasets = [
    (TRAIN_DATA, MANUAL_TRAIN_DATA), (EVAL_DATA, MANUAL_EVAL_DATA)
]
# dataset,manual

for i, data in enumerate(datasets):

    audioset_data_path, manual_data_path = data

    with open(audioset_data_path,'a',newline='') as audioset_data, open(manual_data_path,'r') as manual_data:
        
        csv_writer = csv.writer(audioset_data)

        csv_reader = csv.reader(manual_data)

        headers = next(csv_reader)

        for row in csv_reader:
            food_type = row.pop(1)

            # print(row)
            csv_writer.writerow(row)

        pass
        

    pass

