"""
A convolutional autoencoder for MNIST dataset.
"""
from typing import List, Tuple

import matplotlib.pyplot as plt
import torch
from torch import nn, optim, Tensor
from torchvision import transforms as transforms
from torchinfo import summary

from datasets.mnist import mnist
from examples.mnist_autoencoder import display_reconstructions
from neural_networks.training import train_autoencoder
from utils.assert_shape import AssertShape
from utils.cuda import print_cuda_configuration
from utils.seeds import set_seeds


class ConvAutoencoder(nn.Module):
    def __init__(self) -> None:
        super(ConvAutoencoder, self).__init__()

        # Encoder: Conv2d layers
        self.encoder = nn.Sequential(
            # [B, 1, 28, 28] -> [B, 16, 14, 14]
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(True),
            AssertShape((16, 14, 14)),

            # [B, 16, 14, 14] -> [B, 32, 7, 7]
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(True),
            AssertShape((32, 7, 7)),

            # [B, 32, 7, 7] -> [B, 64, 1, 1]
            nn.Conv2d(32, 64, 7),
            AssertShape((64, 1, 1)),
        )

        # Decoder: ConvTranspose2d layers
        self.decoder = nn.Sequential(
            # [B, 64, 1, 1] -> [B, 32, 7, 7]
            nn.ConvTranspose2d(64, 32, 7),
            nn.ReLU(True),
            AssertShape((32, 7, 7)),


            # [B, 32, 7, 7] -> [B, 16, 14, 14]
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            AssertShape((16, 14, 14)),

            # [B, 16, 14, 14] -> [B, 1, 28, 28]
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),  # To bring the output between [0, 1]
            AssertShape((1, 28, 28)),
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward pass of the autoencoder model.
        Args:
            x: Input tensor of shape (batch_size, 1, 28, 28)

        Returns: Output, latent code

        """
        # Check input shape
        AssertShape((1, 28, 28))(x)

        z = self.encoder(x)
        x_prime: Tensor = self.decoder(z)
        return x_prime, z


def train() -> None:
    set_seeds()
    print_cuda_configuration()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Configuration ===

    # Path to the directory where downloaded data is stored.
    data_path = "./data"

    # Training batch size.
    batch_size = 100

    # Number of passes over the training data.
    num_epochs = 10

    # Learning rate for the optimizer.
    learning_rate = 1e-3

    transform = transforms.Compose([transforms.ToTensor()]) # Converts pixel values in the range [0, 255] to [0, 1].
    train_loader, test_loader = mnist(data_path, batch_size, transform)

    # Initialize the model, loss function, and optimizer
    model = ConvAutoencoder().to(device)
    summary(model, input_size=(batch_size, 1, 28, 28))
    optimizer: optim.Optimizer = optim.Adam(model.parameters(), lr=learning_rate)


    per_batch_loss: List[float] = train_autoencoder(
        model, device, train_loader, optimizer, num_epochs
    )
    print('Training finished.')

    # === Testing ===

    test_data_iter = iter(test_loader)
    test_images, _labels = next(test_data_iter)
    test_images = test_images.cuda()


    (reconstructed, _code) = model(test_images)
    test_images = test_images.cpu()
    reconstructed = reconstructed.cpu()

    # Plot the per-batch training loss.
    plt.figure(1)
    plt.plot(per_batch_loss)
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.title("Per-batch Training Loss")

    # Compare inputs and reconstructed images.
    display_reconstructions(test_images, reconstructed)

    plt.show()


if __name__ == "__main__":
    train()