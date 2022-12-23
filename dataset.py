from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Normalize, ToTensor


def mnist(batch_size: int, shuffle: bool, num_workers: int, train: bool) -> DataLoader:
    """Gets the training data."""

    dataset = MNIST(
        root="dataset",
        train=train,
        download=True,
        transform=Compose([ToTensor(), Normalize(mean=(0.1307,), std=(0.3081,))]),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )

    return dataloader
