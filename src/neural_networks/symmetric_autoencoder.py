# Autoencoder with symmetric or "mirrored" encoder and decoder architecture.

from typing import List, Tuple

import torch.nn as nn
from torch import Tensor


# Autoencoder with symmetric or "mirrored" encoder and decoder architecture.
class SymmetricAutoencoder(nn.Module):
    def __init__(self, layer_sizes: List[int]) -> None:
        """
        An Autoencoder with "mirrored" encoder and decoder architecture.

        The decoder is the mirror of the encoder. The first layer size must be the input size, and the last layer
        size must be the code size. layer_sizes must have at least 2 elements. This ensures that the input size and
        code size are specified.

        :param layer_sizes: List of layer sizes for the encoder.
        """

        super(SymmetricAutoencoder, self).__init__()
        # The input size and code size must be specified.
        if len(layer_sizes) < 2:
            raise ValueError("layer_sizes must have at least 2 elements.")

        # Encoder
        encoder_layers: List[nn.Module] = []
        input_size = layer_sizes[0]
        for i, output_size in enumerate(layer_sizes[1:]):
            encoder_layers.append(nn.Linear(input_size, output_size))
            encoder_layers.append(nn.ReLU())
            input_size = output_size
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers: List[nn.Module] = []
        reversed_layer_sizes = layer_sizes.copy()
        reversed_layer_sizes.reverse()
        input_size = reversed_layer_sizes[0]
        for i, output_size in enumerate(reversed_layer_sizes[1:]):
            decoder_layers.append(nn.Linear(input_size, output_size))
            if i < len(layer_sizes) - 2:  # Add ReLU for all but the last layer
                decoder_layers.append(nn.ReLU())
            input_size = output_size
        # Clamp the range of the output to [-1, 1].
        decoder_layers.append(nn.Tanh())
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward pass of the autoencoder model.

        :param x: Input tensor of shape (batch_size, input_size)
        :return: Output, latent code
        """
        code = self.encoder(x)
        recovered = self.decoder(code)
        return recovered, code
