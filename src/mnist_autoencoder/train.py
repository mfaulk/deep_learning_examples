# Train an autoencoder on the MNIST dataset.

from typing import List, Type

import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import KFold
from torch import Tensor, optim
from torch import nn
from torch.utils.data import DataLoader, Subset, Dataset
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


def k_fold_cross_validation(
        k: int,
        dataset: Dataset,
        model_class: Type[nn.Module],
        device: torch.device,
        criterion: nn.Module,
        batch_size: int,
        learning_rate: float,
        num_epochs: int) -> List[float]:
    """
    Perform k-fold cross-validation.
    :param k: Number of folds.
    :param dataset: Dataset to split into k folds.
    :param model_class: Model class to instantiate.
    :param device: Device to run the model on.
    :param criterion: Loss function.
    :param batch_size: Training batch size.
    :param learning_rate: Learning rate for the optimizer.
    :param num_epochs: Number of training epochs.
    :return: Average validation loss on each fold.
    """

    kfold = KFold(n_splits=k, shuffle=True)

    # Validation loss of each folding.
    validation_losses = []

    for fold, (train_indices, val_indices) in enumerate(kfold.split(dataset)):
        print(f'Fold {fold + 1}')

        train_subset = Subset(dataset, train_indices)
        validation_subset = Subset(dataset, val_indices)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4)
        validation_loader = DataLoader(validation_subset, batch_size=batch_size, shuffle=False, num_workers=4)

        model = model_class().to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        train_autoencoder(model, device, train_loader, optimizer, criterion, num_epochs)
        loss = evaluate_autoencoder(model, device, validation_loader, criterion)
        validation_losses.append(loss)

    return validation_losses


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

    # Number of folds for k-fold cross-validation.
    k_folds = 5

    # === Data ===
    train_loader, test_loader = get_mnist_data(data_path, batch_size)

    # === Training ===
    criterion: nn.MSELoss = nn.MSELoss()

    # Model selection via k-fold cross-validation.
    loss_per_folding = k_fold_cross_validation(
        k_folds, train_loader.dataset, Autoencoder, device, criterion, batch_size, learning_rate, num_epochs)
    print(f'Average Validation Loss: {sum(loss_per_folding) / k_folds:.4f}')

    # Train the final model on the full training data.
    print('Training the final model on the full training data.')
    model: Autoencoder = Autoencoder().cuda()
    summary(model, input_size=(batch_size, 28 * 28))
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
