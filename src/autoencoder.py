# Autoencoder neural network for MNIST dataset.

import torch.nn as nn
from torch import Tensor


# Autoencoder model
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 12),
            # nn.ReLU(True),
            # nn.Linear(12, 3)
        )
        self.decoder = nn.Sequential(
            # nn.Linear(3, 12),
            # nn.ReLU(True),
            nn.Linear(12, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, 28 * 28),
            nn.Tanh()
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the autoencoder model.

        :param x: input tensor of shape (batch_size, 28 * 28)
        :return: Recovered tensor.
        """
        code = self.encoder(x)
        recovered = self.decoder(code)
        return recovered
