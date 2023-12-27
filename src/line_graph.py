import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# example input
# data = [
#     {
#         "time": {"start": 0, "end": 0.96},
#         "tags": [("biting", 0.4), ("chewing", 0.5), ("swallow", 0.1)],
#     },
# ]

image_folder = Path("images")
image_folder.mkdir(exist_ok=True)


example_data = []

for i in range(100):
    probability = np.random.random(3)
    probability /= probability.sum()

    example_data.append(
        {
            "time": {"start": i * 0.1, "end": i * 0.1 + 0.96},
            "tags": [
                ("biting", probability[0]),
                ("chewing", probability[1]),
                ("swallow", probability[2]),
            ],
        }
    )

def plot_data(data,graph_name="only_intake"):

    times = []
    biting_probs, chewing_probs, swallow_probs  = [], [], []
    for segment in data:
        biting_probs.append(segment["tags"][0][1])
        chewing_probs.append(segment["tags"][1][1])
        swallow_probs.append(segment["tags"][2][1])
        times.append(segment["time"]["start"])

    plt.plot(times, biting_probs, label="biting")
    plt.plot(times, chewing_probs, label="chewing")
    plt.plot(times, swallow_probs, label="swallow")

    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Probability")
    plt.title("Time vs Probability")

    plt.show()
    plt.savefig(Path(image_folder,graph_name+"_transcribe_results.png"))
