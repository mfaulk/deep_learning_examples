"""
Train a Variational Autoencoder (VAE) on the MNIST dataset.

This example follows the VAE paper by Kingma and Welling, Auto-Encoding Variational Bayes, ICLR, 2014.
"""

from typing import Tuple

import matplotlib.pyplot as plt
import torch
from torch import nn, optim, Tensor
import torch.nn.functional as F
from torchinfo import summary
from torchvision import transforms as transforms

from datasets.mnist import mnist
from examples.mnist_autoencoder import display_reconstructions
from utils.assert_shape import AssertShape
from utils.cuda import print_cuda_configuration
from utils.seeds import set_seeds


def vae_loss(
    x: torch.Tensor, x_prime: torch.Tensor, mu: torch.Tensor, log_variance: torch.Tensor
) -> torch.Tensor:
    """
    Loss function for the Variational Autoencoder (VAE), assuming a Gaussian prior and approximate posterior.

    :param x: Input data.
    :param x_prime: Reconstructed data.
    :param mu: Mean of the latent code.
    :param log_variance: Log of the variance of the latent code.
    :return: Sum of binary cross-entropy loss and KL divergence loss.
    """
    # Sum of binary cross-entropy loss over all elements of the batch.
    reconstruction_loss: Tensor = F.binary_cross_entropy(x_prime, x, reduction='sum')

    # KL Divergence
    # Appendix B of Kingma and Welling gives an analytical solution for the KL divergence when
    # 1. The prior is the standard normal distribution, i.e. p_{\theta}(z) = N(z; 0, I)
    # 2. The approximate posterior distribution q_{\phi}(z|x) is Gaussian with mean mu and diagonal covariance matrix sigma^2

    # Sum of KL divergence over all elements of the batch.
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_divergence: Tensor = -0.5 * torch.sum(1 + log_variance - mu.pow(2) - log_variance.exp())

    return reconstruction_loss + kl_divergence

class ConvolutionalVAE(nn.Module):
    def __init__(self, code_size: int) -> None:
        super(ConvolutionalVAE, self).__init__()
        self.code_size = code_size

        # Encoder outputs mu and log(sigma^2) of the latent code.
        # log(sigma^2) is used instead of sigma^2 to avoid possible numerical issues with small values.
        self.encoder = nn.Sequential(
            # [B, 1, 28, 28] -> [B, 16, 14, 14]
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(True),

            # [B, 16, 14, 14] -> [B, 32, 7, 7]
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(True),
            AssertShape((32, 7, 7)),

            # [B, 32, 7, 7] -> [B, code_size * 2, 1, 1]
            nn.Conv2d(32, code_size * 2, 7),
            AssertShape((code_size * 2, 1, 1)),
        )

        # Decoder outputs the mean of the output data.
        self.decoder = nn.Sequential(
            # [B, code_size, 1, 1] -> [B, 32, 7, 7]
            nn.ConvTranspose2d(code_size, 32, 7),
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

    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Encode the input data x into the mean and log of the variance of the latent code.

        :param x: Input tensor of shape (batch_size, input_size)
        :return: mu_z, log_variance_z
        """
        mu_z, log_variance_z = self.encoder(x).chunk(2, dim=1)
        return mu_z, log_variance_z

    def sample_z(self, mu_z: Tensor, log_variance_z: Tensor) -> Tensor:
        """
        Sample the latent code z from the mean and log of the variance of the latent code.

        :param mu_z: Mean of the latent code.
        :param log_variance_z: Log of the variance of the latent code.
        :return: Sampled latent code z.
        """
        # sigma = exp(log(sigma)), and log(sigma) = log(sigma^2) / 2
        sigma = torch.exp(0.5 * log_variance_z)
        epsilon = torch.randn_like(sigma)
        z = mu_z + epsilon * sigma
        return z

    def decode(self, z: Tensor) -> Tensor:
        """
        Decode the latent code z into the mean of the output data.

        :param z: Latent code tensor of shape (batch_size, code_size)
        :return: mu_x
        """
        mu_x: Tensor = self.decoder(z)
        return mu_x

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass of the VAE model.

        :param x: Input tensor of shape (batch_size, input_size)
        :return: mu_x, mu_z, log_variance_z
        """

        mu_z, log_variance_z = self.encode(x)
        z = self.sample_z(mu_z, log_variance_z)
        mu_x = self.decode(z)
        return mu_x, mu_z, log_variance_z

    def generate(self, num_samples: int, device) -> Tensor:
        """
        Generate new samples from the VAE model.

        :param num_samples: Number of samples to generate.
        :param device: Device to use.
        :return: Generated samples.
        """

        # Sample noise from a standard normal distribution.
        z = torch.randn(num_samples, self.code_size).to(device)
        # Convert z to a tensor of shape (num_samples, code_size, 1, 1)
        z = z.view(-1, self.code_size, 1, 1)
        samples: Tensor = self.decoder(z)
        return samples

def main() -> None:
    set_seeds()
    print_cuda_configuration()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Configuration ===

    # Path to the directory where downloaded data is stored.
    data_path = "./data"

    # Training batch size.
    batch_size = 100

    # Number of passes over the training data.
    num_epochs = 200

    # Learning rate for the optimizer.
    learning_rate = 1e-3

    latent_code_size = 20

    # === Data ===
    transform = transforms.Compose([transforms.ToTensor()])  # Converts pixel values in the range [0, 255] to [0, 1].
    train_loader, test_loader = mnist(data_path, batch_size, transform)

    vae = ConvolutionalVAE(latent_code_size).to(device)
    summary(vae, input_size=(batch_size, 1, 28, 28))

    # === Training ===

    vae.train()
    vae.to(device)
    optimizer = optim.Adam(vae.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        epoch_loss = 0.0
        for images, _labels in train_loader:
            images = images.to(device)

            # Forward pass
            outputs, mu, sigma = vae(images)
            batch_loss = vae_loss(images, outputs, mu, sigma)
            epoch_loss += batch_loss.item()

            # Backward pass and parameter updates
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

        avg_train_loss = epoch_loss / float(len(train_loader.dataset))
        print(f"  Average Training Loss: {avg_train_loss:.8f}")

    print("Training complete.")

    # === Testing ===

    test_data_iter = iter(test_loader)
    test_images, _labels = next(test_data_iter)
    test_images = test_images.cuda()

    (reconstructed, mu_z, log_variance_z) = vae.forward(test_images)
    # print(f"Reconstructed shape: {reconstructed.shape}")
    # print(f"mu_z shape: {mu_z.shape}")
    # print(f"log_variance_z shape: {log_variance_z.shape}")

    test_images = test_images.cpu()
    reconstructed = reconstructed.cpu()

    # Compare inputs and reconstructed images.
    display_reconstructions(test_images, reconstructed)

    # === Generate samples ===
    num_samples = 100
    vae.to(device)
    samples = vae.generate(num_samples, device).cpu().detach()
    samples = samples.view(num_samples, 28, 28)

    num_row = 10
    num_col = 10

    fig, axes = plt.subplots(num_row, num_col, figsize=(2 * num_col, 2 * num_row))
    for i in range(num_samples):
        ax = axes[i // num_col, i % num_col]
        ax.imshow(samples[i], cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
