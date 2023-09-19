import torch
from torcheval.metrics import WordErrorRate

metric = WordErrorRate()

# TODO
metric.update(["text1", "text2"])

print(metric.compute())
