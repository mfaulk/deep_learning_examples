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
        num_epochs: int
) -> List[float]:
    """
    Train the autoencoder model.
    :param model: Initial model. This model will be modified during training.
    :param device: Device to run the model on.
    :param train_loader: training data loader. Data must be a tuple (inputs, labels).
    :param optimizer: Optimizer for training.
    :param num_epochs: Number of epochs for training.
    :return: Per-batch training loss.
    """

    # Per-batch training loss. This includes batches from all epochs.
    per_batch_loss: List[float] = []

    # === Training ===
    model.train()
    for epoch in range(num_epochs):

        # Average batch training loss in this epoch.
        # TODO: avg loss per sample would be clearer.
        epoch_avg_batch_loss = 0.0

        start_time = time.time()
        # Iterate over training batches.
        for inputs, _labels in train_loader:
            inputs = inputs.to(device)

            # Forward pass
            outputs, _code = model(inputs)
            loss = nn.MSELoss()
            loss = loss(outputs, inputs)
            epoch_avg_batch_loss += loss.item()
            per_batch_loss.append(loss.item())

            # Backward pass and parameter updates
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        elapsed_time = time.time() - start_time
        num_batches = len(train_loader)
        epoch_avg_batch_loss /= num_batches
        print(
            f'  Epoch [{epoch + 1}/{num_epochs}], Average Training Loss per Batch: {epoch_avg_batch_loss:.4f}, Time: {elapsed_time:.2f} seconds')

    return per_batch_loss
