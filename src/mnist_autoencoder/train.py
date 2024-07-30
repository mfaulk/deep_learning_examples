# Train an autoencoder on the MNIST dataset.

import time
from typing import List

import matplotlib.pyplot as plt
import torch
from torch import Tensor, optim
from torch import nn
from torch.utils.data import DataLoader
from torchinfo import summary

from src.mnist_autoencoder.autoencoder import Autoencoder
from src.mnist_autoencoder.load import get_mnist_data
from src.utils.cuda import print_cuda_configuration
from src.utils.seeds import set_seeds


def display_reconstructions(original: Tensor, reconstructed: Tensor, num_display: int = 10):
    """
    Display the original and reconstructed images.
    :param original: Original images.
    :param reconstructed: Reconstructed images.
    :param num_display: Number of original-reconstructed image pairs to display.
    :return:
    """
    fig, axes = plt.subplots(nrows=2, ncols=num_display, sharex=True, sharey=True, figsize=(20, 4))
    for images, row in zip([original, reconstructed], axes):
        for img, ax in zip(images, row):
            ax.imshow(img.view(28, 28).detach().numpy(), cmap='gray')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)


def train(
        model: nn.Module,
        device: torch.device,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        num_epochs: int
) -> List[float]:
    """
    Train the autoencoder model.
    :param model: Initial model. This model will be modified during training.
    :param device: Device to run the model on.
    :param train_loader: training data loader.
    :param optimizer: Optimizer for training.
    :param criterion: Loss function.
    :param num_epochs: Number of epochs for training.
    :return: Per-batch training loss.
    """

    # Per-batch training loss
    per_batch_loss: List[float] = []

    # === Training ===
    loss = 0
    for epoch in range(num_epochs):
        start_time = time.time()
        for inputs, _labels in train_loader:
            inputs = inputs.to(device)

            # Forward pass
            outputs, code = model(inputs)
            loss = criterion(outputs, inputs)
            per_batch_loss.append(loss.item())

            # Backward pass and parameter updates
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        elapsed_time = time.time() - start_time
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Time: {elapsed_time:.2f} seconds')

    return per_batch_loss


def evaluate(
        model: nn.Module,
        device: torch.device,
        val_loader: DataLoader,
        criterion: nn.Module
) -> float:
    """
    Evaluate the reconstruction loss on validation data.
    :param model: NN model.
    :param device: Device to run the model on.
    :param val_loader: DataLoader for validation data.
    :param criterion: Loss function.
    :return: Average loss on validation data.
    """
    model.eval()  # Set model to evaluation mode.
    loss = 0
    with torch.no_grad():  # Disable gradient calculation.
        for inputs, _target in val_loader:
            inputs = inputs.to(device)
            outputs, _codes = model(inputs)
            loss += criterion(outputs, inputs)

    avg_loss = loss / len(val_loader.dataset)
    return avg_loss


def main():
    set_seeds()
    print_cuda_configuration()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Configuration ===

    # Path to the directory where downloaded data is stored.
    data_path = './data'

    # Training batch size.
    batch_size = 100

    # Number of passes over the training data.
    num_epochs = 3

    # Learning rate for the optimizer.
    learning_rate = 1e-3

    # === Model ===
    model: Autoencoder = Autoencoder().cuda()
    summary(model, input_size=(batch_size, 28 * 28))

    # === Data ===
    train_loader, test_loader = get_mnist_data(data_path, batch_size)

    # === Training ===
    criterion: nn.MSELoss = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    per_batch_loss: List[float] = train(model, device, train_loader, optimizer, criterion, num_epochs)

    # === Testing ===
    avg_test_loss = evaluate(model, device, test_loader, criterion)
    print(f'Average Testing Loss: {avg_test_loss:.4f}')

    # === Visualizations ===
    test_data_iter = iter(test_loader)
    test_images, _labels = next(test_data_iter)
    test_images = test_images.cuda()

    (reconstructed, _code) = model.forward(test_images)
    test_images = test_images.cpu()
    reconstructed = reconstructed.cpu()

    # Plot the per-batch training loss.
    plt.figure(1)
    plt.plot(per_batch_loss)
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title('Per-batch Training Loss')

    # Compare inputs and reconstructed images.
    display_reconstructions(test_images, reconstructed)

    plt.show()


if __name__ == '__main__':
    main()
