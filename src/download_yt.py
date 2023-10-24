import utils
import csv
from pathlib import Path
import soundfile

CSV_PATH = Path("datasets/manual_data.csv")
OUTPUT_DIR = Path("datasets/manual")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def format_path(youtube_id, start, end):
    # TODO: floor/ceil instead of int
    return f"{youtube_id}_{int(start)}-{int(end)}.wav"


# utils.get_audio_from_yt(
#     youtube_link=utils.get_yt_url("tOkANpLTqvc"),
#     save_path="datasets/test.wav",
#     start_second=0,
#     end_second=1,
# )
# data, samplerate = soundfile.read("datasets/test.wav")
# soundfile.write("datasets/test.wav", data, samplerate, subtype="PCM_16")

with open(CSV_PATH) as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    for rows in reader:
        youtube_id, start, end = rows[0], float(rows[2]), float(rows[3])

        output_path = OUTPUT_DIR / format_path(youtube_id, start, end)
        if not output_path.exists():
            path = utils.get_audio_from_yt(
                youtube_link=utils.get_yt_url(youtube_id),
                save_path=str(output_path),
                start_second=start,
                end_second=end,
            )["audio_path"]

            # for compatibility with:
            # https://www.tensorflow.org/api_docs/python/tf/audio/decode_wav
            data, samplerate = soundfile.read(path)
            soundfile.write(path, data, samplerate, subtype="PCM_16")

        print(output_path)
