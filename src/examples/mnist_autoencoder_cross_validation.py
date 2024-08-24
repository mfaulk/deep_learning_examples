# Train an autoencoder on the MNIST dataset.

import torch
from torch import nn
from torchvision import transforms as transforms

from datasets.mnist import mnist
from model_selection.configuration_space import generate_configurations
from model_selection.cross_validation import k_fold_cross_validation
from neural_networks.symmetric_autoencoder import SymmetricAutoencoder
from src.utils.cuda import print_cuda_configuration
from src.utils.seeds import set_seeds


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
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        torch.flatten,
    ])
    train_loader, test_loader = mnist(data_path, batch_size, transform)
    image_size = 784 # 28 * 28 pixels.

    # === Training ===
    criterion: nn.MSELoss = nn.MSELoss()

    # Model selection via k-fold cross-validation.

    widths = [150, 500, 1000, 1500, 2000, 2500, 3000, 5000]
    min_depth = 1
    max_depth = 3

    # Generate all possible configurations.
    all_configurations = generate_configurations(min_depth, max_depth, widths)

    # Retain only configurations with final layer of size 150.
    all_configurations = [config for config in all_configurations if config[-1] == 150]

    min_cost = float('inf')
    best_configuration = all_configurations[0]

    for configuration in all_configurations:
        # The first layer size is the image size.
        configuration = [image_size] + configuration
        print(f'Trying configuration: {configuration}')

        def model_factory() -> SymmetricAutoencoder:
            return SymmetricAutoencoder(configuration)

        loss_per_folding = k_fold_cross_validation(
            k_folds, train_loader.dataset, model_factory, device, criterion, batch_size, learning_rate, num_epochs)
        avg_loss = sum(loss_per_folding) / k_folds
        print(f'Average cross validation loss: {avg_loss}')
        if avg_loss < min_cost:
            min_cost = avg_loss
            best_configuration = configuration

    print(f'Best configuration: {best_configuration}\nAverage loss: {min_cost}')

    # TODO: plot the loss for each configuration.

if __name__ == '__main__':
    main()
