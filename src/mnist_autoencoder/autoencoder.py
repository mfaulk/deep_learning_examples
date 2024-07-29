# Autoencoder neural network for MNIST dataset.

import torch.nn as nn
from torch import Tensor


# Autoencoder model. 784 -> 128 -> 64 -> [12] -> 64 -> 128 -> 784
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            # Layer 1: 784 -> 128
            nn.Linear(28 * 28, 128),
            nn.ReLU(True),

            # Layer 2: 128 -> 64
            nn.Linear(128, 64),
            nn.ReLU(True),

            # Layer 3: 64 -> 12
            nn.Linear(64, 12),
        )
        self.decoder = nn.Sequential(
            # Layer 4: 12 -> 64
            nn.Linear(12, 64),
            nn.ReLU(True),

            # Layer 5: 64 -> 128
            nn.Linear(64, 128),
            nn.ReLU(True),

            # Layer 6: 128 -> 784
            nn.Linear(128, 28 * 28),
            nn.Tanh()
        )

    def forward(self, x: Tensor) -> (Tensor, Tensor):
        """
        Forward pass of the autoencoder model.

        :param x: input tensor of shape (batch_size, 28 * 28)
        :return: Recovered tensor, code tensor
        """
        code = self.encoder(x)
        recovered = self.decoder(code)
        return recovered, code
