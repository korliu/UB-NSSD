import utils
import csv
from pathlib import Path

CSV_PATH = Path("datasets/manual_data.csv")
OUTPUT_DIR = Path("datasets/manual")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def format_path(youtube_id, start, end):
    return f"{youtube_id}_{int(start)}-{int(end)}.wav"


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

        print(output_path)