#!/usr/bin/env python
import argparse
import os
import json

import torch
import torch.nn as nn
import torch.optim as optim

import wandb

from dataset import mnist
from model import MNISTModel


# NOTE: Reproducibility
torch.manual_seed(0)
torch.use_deterministic_algorithms(True)


def parse_args() -> argparse.Namespace:
    """Parses the command line arguments."""

    parser = argparse.ArgumentParser("MNIST model training")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--shuffle", action="store_false", help="whether to shuffle the dataset")
    parser.add_argument("--num_workers", type=int, default=8, help="worker threads")
    parser.add_argument("--learning_rate", type=float, default=2e-3, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="weight decay")
    parser.add_argument("--epochs", type=int, default=4, help="number of epochs")
    parser.add_argument("--model_dir", type=str, default="model", help="directory to store models")
    parser.add_argument("--model_name", type=str, default="model.pt", help="model path")

    return parser.parse_args()


def main() -> None:
    """Runs the training loop."""

    args = parse_args()

    wandb.init(project="mnist", config=vars(args))

    dataloader = mnist(
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=args.num_workers,
        train=True,
    )

    model = MNISTModel()
    model = nn.DataParallel(model)  # type: ignore
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    for epoch in range(args.epochs):
        running_loss = 0.0
        for idx, (features, labels) in enumerate(dataloader):
            # move features and labels onto the device
            features = features.cuda()
            labels = labels.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # accumulate loss and log
            running_loss += loss.item()
            if idx % 375 == 374:
                print(f"Epoch {epoch} | Steps: {idx + 1:<4} | Loss: {running_loss / 375:.3f}")
                wandb.log({"epoch": epoch, "steps": idx + 1, "loss": round(running_loss / 375, 3)})
                running_loss = 0.0

    # Create the directory for storing models if it does not already exist
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    # Save the model
    torch.save(model.module.state_dict(), os.path.join(args.model_dir, args.model_name))

    # Save the parameters
    with open(os.path.join(args.model_dir, "params.json"), "w") as file:
        json.dump(vars(args), file, indent=4)


if __name__ == "__main__":
    main()
