from pathlib import Path
import os
import numpy as np
import csv
from random import sample
import librosa
import soundfile as sf
import random

spliced_output_dir = Path("datasets","spliced_audio")

# food intake
intake_audios_csv = Path("datasets","food_intake_data.csv")

intake_audios = []

with open(intake_audios_csv, 'r') as f:

    csv_reader = csv.reader(f)

    headers = next(csv_reader)

    path,variant,food,sex,other_labels,source = list(range(0,len(headers)))

    for i, data in enumerate(csv_reader):

        audio_data = data[path], data[variant]

        intake_audios.append(audio_data)

intake_audio_dir = Path(spliced_output_dir,"food_intake")
intake_audio_dir.mkdir(parents=True, exist_ok=True)

with open(intake_audio_dir.as_posix()+"food_intake.csv", 'w+') as f:
    csv_writer = csv.writer(f)
    
    headers = ["audio_path", "sounds_in_order"]
    csv_writer.writerow(headers)
    
    max_samples = 5

    random.seed(42)
    random.shuffle(intake_audios)
    
    splice_index = 0
    audio_index = 0
    while audio_index < len(intake_audios):
        n_audios = np.random.choice(np.arange(2,max_samples+1))

        start = audio_index
        end = min((audio_index+n_audios),len(intake_audios))

        audio_clips = intake_audios[start:end]

        spliced_audio = []
        sounds_in_order = []
        for clip_data in audio_clips:
            clip_path, clip_sound = clip_data
            audio_wav, sr = librosa.load(clip_path)

            spliced_audio.append(audio_wav)
            sounds_in_order.append(clip_sound)


        spliced_audio = np.concatenate(spliced_audio)

        splice_index += 1
        output_path = Path(intake_audio_dir, f"splice_{splice_index}.wav")
        sf.write(output_path, data=spliced_audio, samplerate=8000)

        csv_data = [output_path.as_posix(), ",".join(sounds_in_order)]
        csv_writer.writerow(csv_data)

        audio_index += n_audios

        
        


# youtube