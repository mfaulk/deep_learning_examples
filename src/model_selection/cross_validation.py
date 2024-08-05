from typing import Callable, List

import torch
from sklearn.model_selection import KFold
from torch import nn, optim
from torch.utils.data import Dataset, Subset, DataLoader

from evaluating import evaluate_autoencoder
from neural_networks.symmetric_autoencoder import SymmetricAutoencoder
from training import train_autoencoder


def k_fold_cross_validation(
        k: int,
        dataset: Dataset,
        model_factory: Callable[[], SymmetricAutoencoder],
        device: torch.device,
        criterion: nn.Module,
        batch_size: int,
        learning_rate: float,
        num_epochs: int) -> List[float]:
    """
    Perform k-fold cross-validation.
    :param k: Number of folds.
    :param dataset: Dataset to split into k folds.
    :param model_factory: Factory function for the model.
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
        print(f'Fold {fold}')

        train_subset = Subset(dataset, train_indices)
        validation_subset = Subset(dataset, val_indices)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4)
        validation_loader = DataLoader(validation_subset, batch_size=batch_size, shuffle=False, num_workers=4)

        model = model_factory().to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        train_autoencoder(model, device, train_loader, optimizer, num_epochs)
        loss = evaluate_autoencoder(model, device, validation_loader, criterion)
        validation_losses.append(loss)

    return validation_losses
