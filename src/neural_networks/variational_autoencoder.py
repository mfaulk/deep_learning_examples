# Variational Autoencoder (VAE) class.
#
# The Variational Autoencoder (VAE) is a generative model that learns a probabilistic latent
# representation of the input data. The encoder outputs the mean and log of the variance of the latent code.
# and the decoder samples from the distribution defined by the mean and variance to generate the output.
# The VAE is trained to minimize the reconstruction error and the Kullback-Leibler (KL) divergence between
# the learned distribution and the prior distribution of the latent code.

from typing import List, Tuple

import torch
import torch.nn as nn
from torch import Tensor


class VariationalAutoencoder(nn.Module):
    def __init__(self, input_size: int, code_size: int) -> None:
        """
        Variational Autoencoder (VAE).

        :param input_size: Dimension of input data x.
        :param code_size: Dimension of latent code z.
        """
        super(VariationalAutoencoder, self).__init__()

        self.code_size = code_size

        # Encoder computes the mean and log of the variance of the latent code.
        encoder_layers: List[nn.Module] = [
            nn.Linear(input_size, input_size),
            nn.ReLU(),
            # Outputs: mu and log(sigma^2) of the latent code.
            # log(sigma^2) is used instead of sigma^2 to avoid possible numerical issues with small values.
            nn.Linear(input_size, code_size * 2),
        ]

        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder computes the mean of the output data.
        decoder_layers: List[nn.Module] = [
            nn.Linear(code_size, input_size),
            nn.ReLU(),
            nn.Linear(input_size, input_size),
            # Pixel outputs must be in the range [0, 1].
            # nn.Sigmoid(),
            nn.Tanh(),
        ]
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass of the VAE model.

        :param x: Input tensor of shape (batch_size, input_size)
        :return: mu_x, mu_z, sigma_z
        """

        # Encode the input data x into the mean and log of the variance of the latent code.
        mu_z, log_variance_z = self.encoder(x).chunk(2, dim=1)

        # log(sigma^2) / 2 = log(sigma), sigma = exp(log(sigma))
        sigma = torch.exp(0.5 * log_variance_z)
        epsilon = torch.randn_like(sigma)

        # z|x ~ N(mu, sigma^2)
        z = mu_z + epsilon * sigma

        # Decode the latent code z into the mean of the output data.
        mu_x = self.decoder(z)

        return mu_x, mu_z, sigma

    def sample(self, num_samples: int, device) -> Tensor:
        """
        Generate samples from the VAE model.

        :param num_samples: Number of samples to generate.
        :return: Generated samples.
        """
        z = torch.randn(num_samples, self.code_size)
        z = z.to(device)
        samples: Tensor = self.decoder(z)
        return samples
