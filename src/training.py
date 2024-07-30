import time
from typing import List

import torch
from torch import nn
from torch.utils.data import DataLoader


def train_autoencoder(
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
    :param train_loader: training data loader. Data must be a tuple (inputs, labels).
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
