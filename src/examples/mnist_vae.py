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
    # 2. The approximate posterior distribution q_{\phi}(z|x) is Gaussian.

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

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass of the VAE model.

        :param x: Input tensor of shape (batch_size, input_size)
        :return: mu_x, mu_z, log_variance_z
        """

        batch_size = x.size(0)

        # Encode the input data x into the mean and log of the variance of the latent code.
        mu_z, log_variance_z = self.encoder(x).chunk(2, dim=1)
        # z = self.sample(batch_size, mu_z, log_variance_z, x.device)

        # log(sigma^2) / 2 = log(sigma), sigma = exp(log(sigma))
        sigma = torch.exp(0.5 * log_variance_z)
        epsilon = torch.randn_like(sigma)

        # z|x ~ N(mu, sigma^2)
        z = mu_z + epsilon * sigma

        # Decode the latent code z into the mean of the output data.
        mu_x = self.decoder(z)

        return mu_x, mu_z, log_variance_z

    def sample(self, num_samples: int, mu_z: Tensor, log_variance: Tensor, device) -> Tensor:
        """
        Generate samples from the VAE model.

        :param num_samples: Number of samples to generate.
        :param mu_z: Mean of the latent code.
        :param log_variance: Log of the variance of the latent code.
        :return: Generated samples.
        """
        sigma = torch.exp(0.5 * log_variance)
        epsilon = torch.randn_like(sigma)

        # z|x ~ N(mu, sigma^2)
        z = mu_z + epsilon * sigma
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
    num_epochs = 10

    # Learning rate for the optimizer.
    learning_rate = 1e-3

    # === Data ===
    transform = transforms.Compose(
        [
            transforms.ToTensor(), # Converts pixel values in the range [0, 255] to [0, 1].
            torch.flatten,
        ]
    )
    transform = transforms.Compose([transforms.ToTensor()]) # Converts pixel values in the range [0, 255] to [0, 1].
    train_loader, test_loader = mnist(data_path, batch_size, transform)

    vae = ConvolutionalVAE(12).to(device)
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
            loss = vae_loss(images, outputs, mu, sigma)
            epoch_loss += loss.item() * batch_size

            # Backward pass and parameter updates
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_train_loss = epoch_loss / float(len(train_loader.dataset))
        print(f"  Average Training Loss: {avg_train_loss:.8f}")

    print("Training complete.")

    # === Testing ===

    test_data_iter = iter(test_loader)
    test_images, _labels = next(test_data_iter)
    test_images = test_images.cuda()

    (reconstructed, mu_z, log_variance_z) = vae.forward(test_images)
    test_images = test_images.cpu()
    reconstructed = reconstructed.cpu()

    # Compare inputs and reconstructed images.
    display_reconstructions(test_images, reconstructed)

    # === Generate samples ===
    num_samples = 100
    samples = vae.sample(num_samples, mu_z, log_variance_z, device).cpu().detach()
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
