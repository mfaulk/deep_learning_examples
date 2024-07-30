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


def testing_loss(autoencoder: Autoencoder, test_loader: DataLoader, mse: nn.MSELoss) -> float:
    """
    Compute the average testing loss.
    :param autoencoder: Autoencoder model.
    :param test_loader: DataLoader for testing data.
    :param mse: Mean Squared Error loss function.
    :return: Testing loss.
    """
    test_loss = 0.0

    num_test_examples = len(test_loader)
    autoencoder.eval()  # Set model to evaluation mode.
    with torch.no_grad():  # Disable gradient calculation.
        for img_batch, _labels in test_loader:
            # img_batch is a tensor of shape (batch_size, 784)
            img_batch = img_batch.cuda()  # Move to GPU

            # Forward pass
            output, code = autoencoder(img_batch)
            loss = mse(output, img_batch)
            test_loss += loss.item()

    return test_loss / num_test_examples


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


def train(model: Autoencoder, train_loader: DataLoader, optimizer, criterion, num_epochs: int) -> List[float]:
    """
    Train the autoencoder model.
    :param model: Initial model. This model will be modified during training.
    :param train_loader: training data loader.
    :param optimizer: Optimizer for training.
    :param criterion: Loss function.
    :param num_epochs: Number of epochs for training.
    :return: Per-batch training loss.
    """
    # loss_fn: nn.MSELoss = nn.MSELoss()
    # optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # Per-batch training loss
    per_batch_loss: List[float] = []

    # === Training ===
    for epoch in range(num_epochs):
        start_time = time.time()  # Start time for the epoch
        for inputs, _labels in train_loader:
            inputs = inputs.cuda()  # Move batch to GPU

            # Forward pass
            outputs, code = model(inputs)
            loss = criterion(outputs, inputs)
            per_batch_loss.append(loss.item())

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        elapsed_time = time.time() - start_time
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Time: {elapsed_time:.2f} seconds')

    return per_batch_loss


def main():
    set_seeds()
    print_cuda_configuration()

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

    # === Training ===
    train_loader, test_loader = get_mnist_data(data_path, batch_size)
    criterion: nn.MSELoss = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    per_batch_loss: List[float] = train(model, train_loader, optimizer, criterion, num_epochs)

    # === Testing ===
    avg_test_loss = testing_loss(model, test_loader, criterion)
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
