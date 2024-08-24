from typing import Tuple

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms as transforms
from torchvision.datasets import MNIST
from typing import Callable, Optional


def mnist(
    data_dir: str, batch_size: int, transform: Optional[Callable] = None
) -> Tuple[DataLoader, DataLoader]:
    """
    Get MNIST Training and Testing data sets.
    Args:
        data_dir:
        batch_size:
        transform:

    Returns:
        (train_loader, test_loader)
    """

    if transform is None:
        transform = transforms.Compose([transforms.ToTensor()])

    training_set = MNIST(root=data_dir, train=True, download=True, transform=transform)
    train_loader = DataLoader(
        training_set, batch_size=batch_size, shuffle=True, num_workers=4
    )

    test_set = torchvision.datasets.MNIST(
        root=data_dir, train=False, download=True, transform=transform
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=4
    )

    return train_loader, test_loader


# def get_mnist_data(data_dir: str, batch_size: int) -> Tuple[DataLoader, DataLoader]:
#     """
#     Get MNIST Training and Testing data sets.
#     :param data_dir: Directory where downloaded data is stored.
#     :param batch_size: Batch size for training and testing data.
#     :return: (train_loader, test_loader)
#     """
#
#     # Transform to normalize pixel values and flatten images to 1D tensor.
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.5,), (0.5,)),
#         torch.flatten,
#     ])
#
#     trainset = MNIST(root=data_dir, train=True, download=True, transform=transform)
#     print(f"Number of MNIST training examples: {len(trainset)}")
#     trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
#
#     testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
#     print(f"Number of MNIST testing examples: {len(testset)}")
#     testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)
#
#     return trainloader, testloader
