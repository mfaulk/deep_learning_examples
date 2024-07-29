import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder


def load_celeba(path: str, batch_size: int) -> (DataLoader, DataLoader):
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
    num_images: int = len(full_dataset)
    print(f"Number of CelebA examples: {num_images}")

    # Split the dataset into training and testing sets
    train_size: int = int(0.8 * num_images)  # 80% for training
    test_size: int = num_images - train_size  # 20% for testing
    print(f"Number of CelebA training examples: {train_size}")
    print(f"Number of CelebA testing examples: {test_size}")
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    # Define the DataLoader
    training_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=6)
    testing_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=6)

    # #
    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     # Normalize pixel values.
    #     transforms.Normalize(mean=[0.5, 0.5, 0.5],
    #                          std=[0.5, 0.5, 0.5]),
    #     torch.flatten,
    # ])
    #
    # # Load the CelebA dataset
    # training_dataset = datasets.CelebA(root=data_path, split='train', download=True, transform=transform)
    # training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
    #
    # testing_dataset = datasets.CelebA(root=data_path, split='test', download=True, transform=transform)
    # testing_dataloader = DataLoader(testing_dataset, batch_size=batch_size, shuffle=False)

    return training_dataloader, testing_dataloader
