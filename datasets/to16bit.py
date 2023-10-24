from pathlib import Path

import soundfile

for f in Path(".").rglob("*.wav"):
    path = f
    print(path)

    data, samplerate = soundfile.read(path)
    soundfile.write(path, data, samplerate, subtype="PCM_16")
