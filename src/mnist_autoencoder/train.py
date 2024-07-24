# Train an autoencoder on the MNIST dataset.

import time

import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch import Tensor
from torch import nn
from torch.utils.data import DataLoader

from src.mnist_autoencoder.autoencoder import Autoencoder
from src.mnist_autoencoder.load import get_mnist_data
from src.utils.cuda import print_cuda_configuration
from src.utils.seeds import set_seeds


def testing_loss(autoencoder: Autoencoder, test_loader: DataLoader, mse: nn.MSELoss) -> float:
    '''
    Compute the average testing loss.
    :param autoencoder: Autoencoder model.
    :param test_loader: DataLoader for testing data.
    :param mse: Mean Squared Error loss function.
    :return: Testing loss.
    '''
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
    '''
    Display the original and reconstructed images.
    :param original: Original images.
    :param reconstructed: Reconstructed images.
    :param num_display: Number of original-reconstructed image pairs to display.
    :return:
    '''
    fig, axes = plt.subplots(nrows=2, ncols=num_display, sharex=True, sharey=True, figsize=(20, 4))
    for images, row in zip([original, reconstructed], axes):
        for img, ax in zip(images, row):
            ax.imshow(img.view(28, 28).detach().numpy(), cmap='gray')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)


def main():
    print_cuda_configuration()

    # Set seeds for reproducibility
    set_seeds()

    train_loader, test_loader = get_mnist_data('./data', batch_size=100)

    autoencoder: Autoencoder = Autoencoder().cuda()

    # Loss function
    mse: nn.MSELoss = nn.MSELoss()

    optimizer = optim.Adam(autoencoder.parameters(), lr=1e-3)

    lambda_l2 = 1e-5  # Regularization strength

    # Training
    num_epochs = 10
    for epoch in range(num_epochs):
        start_time = time.time()  # Start time for the epoch
        for img_batch, _labels in train_loader:
            img_batch = img_batch.cuda()  # Move batch to GPU

            # Forward pass
            output, code = autoencoder(img_batch)
            loss = mse(output, img_batch)

            # L2 regularization on the code vector
            l2_reg = lambda_l2 * torch.norm(code, p=2)
            loss += l2_reg

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Time: {elapsed_time:.2f} seconds')

    # Testing loss
    avg_test_loss = testing_loss(autoencoder, test_loader, mse)
    print(f'Average Testing Loss: {avg_test_loss:.4f}')

    test_data_iter = iter(test_loader)
    test_images, _labels = next(test_data_iter)
    test_images = test_images.cuda()

    (reconstructed, _code) = autoencoder.forward(test_images)

    test_images = test_images.cpu()
    reconstructed = reconstructed.cpu()

    # Visualization
    display_reconstructions(test_images, reconstructed)
    plt.show()  # Display all open figures


if __name__ == '__main__':
    main()
