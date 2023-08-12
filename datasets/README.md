Dataset downloads:

- VocalSound:
    - `https://github.com/YuanGongND/vocalsound#Download-VocalSound`
    - download link: `https://www.dropbox.com/s/c5ace70qh1vbyzb/vs_release_16k.zip?dl=1`: this is the 16kHz version of VocalSound
- PodcastFillers: 
    - `https://drive.google.com/file/d/1fc3LTMt_VqxzlInS7cdP3I2Rarw-8Nlr/view?usp=sharing`
- Food Intake Dataset:
    - `https://www.dropbox.com/sh/t9guqqte7xbqqcz/AABZB1svTYCHa9DzxsuYSLtna?dl=0`
    - `https://drive.google.com/file/d/1v04UxNYPvdfG6rPZcZEuZD1sATYuMnTJ/view?usp=drive_link`: cleaned json from original data
    - Format: {
        "speaker_info": {
            "id": "AA",
            "height_cm": [000],
            "weight_kg": [000],
            "gender": "[Male/Female]",
            "age": 000,
            "ethnicity": "[RACE]"
        },
        "food_type": "FOOD",
        "audio_sampling_frequency": "8000",
        "n_events": 1111,
        "event2label": {
            "bite": "1",
            "chew": "2",
            "swallow": "3",
            "undefined": "4"
        },
        "label2event": {
            "1": "bite",
            "2": "chew",
            "3": "swallow",
            "4": "undefined"
        },
        "events_headers": ["event_number","event_class","start_time_sec", "end_time_sec"],
        "events": [
            
        ]
    }
