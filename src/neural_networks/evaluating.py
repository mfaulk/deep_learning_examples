import torch
from torch import nn
from torch.utils.data import DataLoader


def evaluate_autoencoder(
    model: nn.Module, device: torch.device, val_loader: DataLoader, criterion: nn.Module
) -> float:
    """
    Evaluate the reconstruction loss on validation data.
    :param model: NN model.
    :param device: Device to run the model on.
    :param val_loader: DataLoader for validation data. Data must be a tuple (inputs, labels).
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
