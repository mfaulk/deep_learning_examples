# Train an autoencoder on the MNIST dataset.

from typing import List

import matplotlib.pyplot as plt
import torch
from torch import Tensor, optim
from torch import nn
from torchinfo import summary

from datasets.mnist import get_mnist_data
from model_selection.configuration_space import generate_configurations
from model_selection.cross_validation import k_fold_cross_validation
from neural_networks.evaluating import evaluate_autoencoder
from neural_networks.symmetric_autoencoder import SymmetricAutoencoder
from neural_networks.training import train_autoencoder
from src.utils.cuda import print_cuda_configuration
from src.utils.seeds import set_seeds


def display_reconstructions(original: Tensor, reconstructed: Tensor, num_display: int = 5) -> None:
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


def main() -> None:
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

    # Number of folds for k-fold cross-validation.
    k_folds = 3

    # === Data ===
    train_loader, test_loader = get_mnist_data(data_path, batch_size)

    # === Training ===
    criterion: nn.MSELoss = nn.MSELoss()

    # Model selection via k-fold cross-validation.

    widths = [32, 64, 128, 256]
    min_depth = 1
    max_depth = 1

    # Generate all possible configurations.
    all_configurations = generate_configurations(min_depth, max_depth, widths)
    min_cost = float('inf')
    best_configuration = all_configurations[0]

    for configuration in all_configurations:
        # The input layer size is 784 = (28 * 28).
        configuration = [784] + configuration
        print(f'Trying configuration: {configuration}')

        def model_factory() -> SymmetricAutoencoder:
            return SymmetricAutoencoder(configuration)

        loss_per_folding = k_fold_cross_validation(
            k_folds, train_loader.dataset, model_factory, device, criterion, batch_size, learning_rate, num_epochs)
        avg_loss = sum(loss_per_folding) / k_folds
        if avg_loss < min_cost:
            min_cost = avg_loss
            best_configuration = configuration

    # === Training the final model ===

    print('\nTraining the final model on the full training data.')
    model: SymmetricAutoencoder = SymmetricAutoencoder(best_configuration).cuda()
    summary(model, input_size=(batch_size, 28 * 28))
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    per_batch_loss: List[float] = train_autoencoder(model, device, train_loader, optimizer, num_epochs)

    # === Testing ===

    avg_test_loss = evaluate_autoencoder(model, device, test_loader, criterion)
    print(f'\nAverage Testing Loss: {avg_test_loss:.4f}')

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
