# Autoencoder for the CelebA image dataset.

import torch.nn as nn
from torch import Tensor


# Autoencoder model.
#
# CelebA images are 218 x 178 x 3 = 116_412 pixels
# 116_412 -> [320] -> 116_412
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            # Layer 1: 116_412 -> 500
            nn.Linear(218 * 178 * 3, 500),
            nn.ReLU(True),
            # Layer 2: 500 -> 320
            nn.Linear(500, 320),
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            # Layer 1: 320 -> 116_412
            nn.Linear(320, 218 * 178 * 3),
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
