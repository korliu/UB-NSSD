import matplotlib.pyplot as plt
import numpy as np

# example input
# data = [
#     {
#         "time": {"start": 0, "end": 0.96},
#         "tags": [("biting", 0.4), ("chewing", 0.5), ("swallow", 0.1)],
#     },
# ]

data = []

for i in range(100):
    probability = np.random.random(3)
    probability /= probability.sum()

    data.append(
        {
            "time": {"start": i * 0.1, "end": i * 0.1 + 0.96},
            "tags": [
                ("biting", probability[0]),
                ("chewing", probability[1]),
                ("swallow", probability[2]),
            ],
        }
    )


x1, x2, x3 = [], [], []
y = []
for segment in data:
    x1.append(segment["tags"][0][1])
    x2.append(segment["tags"][1][1])
    x3.append(segment["tags"][2][1])
    y.append(segment["time"]["start"])

plt.plot(y, x1, label="biting")
plt.plot(y, x2, label="chewing")
plt.plot(y, x3, label="swallow")

plt.legend()
plt.xlabel("Time")
plt.ylabel("Probability")
plt.title("Time vs Probability")

plt.show()
