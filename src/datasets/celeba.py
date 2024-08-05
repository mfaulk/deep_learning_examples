from typing import Tuple

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder


def load_celeba(path: str, batch_size: int) -> Tuple[DataLoader, DataLoader]:
    """
    Get CelebA Training and Testing data sets. Each image has size 218 x 178 x 3.
    :param path: Path to a directory of images.
    :param batch_size: Batch size for training.
    :return: (train_loader, test_loader)
    """

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
        torch.flatten
    ])

    # Load the full dataset
    full_dataset: ImageFolder = ImageFolder(root=path, transform=transform)
    
    # Split the dataset into training and testing sets
    train = 0.8
    test = 0.2
    train_dataset, test_dataset = random_split(full_dataset, [train, test])
    print(f"Split CelebA dataset into {len(train_dataset):_} training and {len(test_dataset):_} testing examples.")

    # Create DataLoaders
    training_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=6)
    testing_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=6)

    return training_dataloader, testing_dataloader
