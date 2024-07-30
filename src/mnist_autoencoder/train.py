# Train an autoencoder on the MNIST dataset.

from typing import List

import matplotlib.pyplot as plt
import torch
from torch import Tensor, optim
from torch import nn
from torchinfo import summary

from evaluating import evaluate_autoencoder
from src.mnist_autoencoder.autoencoder import Autoencoder
from src.mnist_autoencoder.load import get_mnist_data
from src.utils.cuda import print_cuda_configuration
from src.utils.seeds import set_seeds
from training import train_autoencoder


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

    per_batch_loss: List[float] = train_autoencoder(model, device, train_loader, optimizer, criterion, num_epochs)

    # === Testing ===
    avg_test_loss = evaluate_autoencoder(model, device, test_loader, criterion)
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
