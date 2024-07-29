# Autoencoder for the CelebA image dataset.

import torch.nn as nn
from torch import Tensor


# Autoencoder model. 4096 * 3-> [32] -> 4096 * 3
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            # Layer 1: 4096 -> 32
            nn.Linear(64 * 64 * 3, 32),
            nn.ReLU(True),
        )
        self.decoder = nn.Sequential(
            # Layer 2: 32 -> 4096
            nn.Linear(32, 64 * 64 * 3),
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
