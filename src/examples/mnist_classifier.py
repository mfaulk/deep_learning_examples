"""
MNIST Classifier using a Convolutional Neural Network.
"""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary

import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms as transforms
from torchvision.datasets import MNIST

from utils.seeds import set_seeds


# A Convolutional Neural Network for digit classification.
class CNN(nn.Module):
    def __init__(self) -> None:
        super(CNN, self).__init__()

        layers: List[nn.Module] = [
            # Convolutional Layers
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            # Fully Connected Layers
            nn.Flatten(),
            nn.Linear(in_features=64 * 7 * 7, out_features=128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(in_features=128, out_features=10),
            # nn.Softmax(1),
        ]
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the forward pass of the CNN.

        Args:
            x (torch.Tensor): Input tensor representing a batch of images.
                              Expected shape is (batch_size, 1, 28, 28).

        Returns:
            torch.Tensor: The output logits of the network, which have not
                          been passed through a softmax function. The shape
                          of the output tensor is (batch_size, 10), where each
                          element corresponds to the logits for each class.
        """
        x = self.network(x)
        return x


def train(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    epochs: int = 5,
) -> Tuple[List[float], List[float]]:
    """
    Trains the neural network and evaluates it on the test set after each epoch.

    Args:
        model: The neural network model to train.
        criterion: The loss function.
        optimizer: The optimization algorithm.
        train_loader: DataLoader for the training set.
        test_loader: DataLoader for the test set.
        epochs: Number of epochs to train. Defaults to 5.

    Returns:
        Tuple[List[float], List[float]]: Lists of average training and testing losses for each epoch.
    """

    # Per-epoch average training losses.
    train_losses: List[float] = []

    # Per-epoch testing losses.
    test_losses: List[float] = []

    for epoch in range(epochs):
        model.train()  # Set the model to training mode.
        epoch_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # running_train_loss += loss.item() * images.size(0)  # Accumulate loss
            epoch_loss += loss.item()  # Accumulate loss

        # Calculate the average training loss for the epoch
        avg_train_loss = epoch_loss / float(len(train_loader.dataset))
        train_losses.append(avg_train_loss)

        # Evaluate the model on the test set.
        model.eval()  # Set the model to evaluation mode
        epoch_test_loss = 0.0
        with torch.no_grad():  # Disable gradient calculation for evaluation
            for images, labels in test_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                epoch_test_loss += loss.item()

        # Calculate the average test loss for the epoch
        avg_test_loss = epoch_test_loss / float(len(test_loader.dataset))
        test_losses.append(avg_test_loss)

        print(
            f"Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}"
        )

    return train_losses, test_losses


def test(
    model: nn.Module,
    criteria: nn.Module,
    device: torch.device,
    test_loader: torch.utils.data.DataLoader,
) -> None:
    model.eval()
    test_loss: float = 0
    num_test_points = len(test_loader.dataset)
    num_correct: int = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            test_loss += criteria(output, labels).item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[
                1
            ]  # get the index of the max log-probability
            num_correct += pred.eq(labels.view_as(pred)).sum().item()

    test_loss /= float(num_test_points)
    print(
        f"\nTest set: Average loss: {test_loss:.4f},"
        f" Accuracy: {num_correct}/{num_test_points}"
        f" ({100.0 * num_correct / num_test_points:.0f}%)\n"
    )


def main() -> None:
    set_seeds()

    # Path to data directory, relative to project root.
    # If present, data in the MNIST directory will be used. Otherwise, it will be downloaded.
    data_path = "./data"

    # Training batch size.
    batch_size = 128

    # Per-image transformation
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    train_set = MNIST(root=data_path, train=True, download=True, transform=transform)
    test_set = torchvision.datasets.MNIST(
        root=data_path, train=False, download=True, transform=transform
    )

    # Inspect the dataset
    print(f"Train Dataset Size: {len(train_set)}")
    print(f"Test Dataset Size: {len(test_set)}")
    example_image, example_label = train_set[0]
    print(f"Image Shape: {example_image.shape}")  # Should print torch.Size([1, 28, 28])

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=4
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=4
    )

    device = torch.device("cpu")
    model = CNN().to(device)
    summary(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = 5

    # Run Training, Evaluation, and Plotting
    print("Training...")
    _train_losses, _test_losses = train(
        model, criterion, optimizer, train_loader, test_loader, epochs=num_epochs
    )

    print("Testing...")
    test(model, criterion, device, test_loader)


if __name__ == "__main__":
    main()
