import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms as transforms
from torchvision.datasets import MNIST


def get_mnist_data(data_dir: str, batch_size: int) -> (DataLoader, DataLoader):
    '''
    Get MNIST Training and Testing data sets.
    :param data_dir: Directory where downloaded data is stored.
    :param batch_size: Batch size for training and testing data.
    :return: (train_loader, test_loader)
    '''

    # Transform to normalize pixel values and flatten images to 1D tensor.
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        torch.flatten,
    ])

    trainset = MNIST(root=data_dir, train=True, download=True, transform=transform)
    print(f"Number of MNIST training examples: {len(trainset)}")
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    print(f"Number of MNIST testing examples: {len(testset)}")
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    return trainloader, testloader
