import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def load_celeba(data_path: str, batch_size: int) -> (DataLoader, DataLoader):
    '''
    Get CelebA Training and Testing data sets.
    :param data_path: Path to local data cache.
    :param batch_size: Batch size for training.
    :return: (train_loader, test_loader)
    '''

    #
    transform = transforms.Compose([
        transforms.ToTensor(),
        # Normalize pixel values.
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
        torch.flatten,
    ])

    # Load the CelebA dataset
    training_dataset = datasets.CelebA(root=data_path, split='train', download=True, transform=transform)
    training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)

    testing_dataset = datasets.CelebA(root=data_path, split='test', download=True, transform=transform)
    testing_dataloader = DataLoader(testing_dataset, batch_size=batch_size, shuffle=False)

    return training_dataloader, testing_dataloader
