#!/usr/bin/env python
import json

import torch

from dataclasses import dataclass

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from dataset import mnist
from model import MNISTModel


@dataclass
class Params:
    batch_size: int
    shuffle: bool
    num_workers: int
    learning_rate: float
    weight_decay: float
    epochs: int
    model_path: str


def load_params(path: str) -> Params:
    """Loads the parameters from the JSON file."""

    with open(path, "r") as file:
        params = json.load(file)

    return Params(**params)


def main() -> None:
    """Evaluates the model."""

    params = load_params("./params.json")

    dataloader = mnist(
        batch_size=params.batch_size,
        shuffle=params.shuffle,
        num_workers=params.num_workers,
        train=False,
    )

    model = MNISTModel()
    model.load_state_dict(torch.load(params.model_path))
    model.eval()

    actual = []
    pred = []
    with torch.no_grad():
        for features, labels in dataloader:
            features = features.cuda()
            labels = labels.cuda()

            out = model(features)

            # NOTE: No need to apply softmax as it will not change the classification results
            _, predicted = torch.max(out.data, 1)

            pred.extend(predicted.flatten().tolist())
            actual.extend(labels.flatten().tolist())

    accuracy = accuracy_score(actual, pred)
    precision, recall, fscore, _ = precision_recall_fscore_support(actual, pred, average="macro")

    print(f"{accuracy=:.3f}")
    print(f"{precision=:.3f}")
    print(f"{recall=:.3f}")
    print(f"{fscore=:.3f}")


if __name__ == "__main__":
    main()
