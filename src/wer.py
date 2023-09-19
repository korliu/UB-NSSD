import torch
from torcheval.metrics import WordErrorRate

metric = WordErrorRate()

# TODO
# metric.update("text1", "text2")

# print(metric.compute())


def compute_wer(expected_text: str, predicted_text: str):

    metric = WordErrorRate()

    metric.update(expected_text, predicted_text)

    return metric.compute()
