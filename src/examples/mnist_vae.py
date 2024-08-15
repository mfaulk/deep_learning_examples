"""
Train a Variational Autoencoder (VAE) on the MNIST dataset.

This example follows the VAE paper by Kingma and Welling, Auto-Encoding Variational Bayes, ICLR, 2014.
"""

import torch
from torch import nn, optim

from datasets.mnist import get_mnist_data
from neural_networks.variational_autoencoder import VariationalAutoencoder
from utils.cuda import print_cuda_configuration
from utils.seeds import set_seeds


def loss_function(x: torch.Tensor, x_prime: torch.Tensor, mu: torch.Tensor, log_variance: torch.Tensor) -> torch.Tensor:
    """
    Loss function for the Variational Autoencoder (VAE), assuming a Gaussian prior and approximate posterior.

    :param x: Input data.
    :param x_prime: Reconstructed data.
    :param mu: Mean of the latent code.
    :param log_variance: Log of the variance of the latent code.
    :return:
        | ||
        || |_
    """
    # Reconstruction loss.
    reconstruction_loss = nn.functional.binary_cross_entropy(x_prime, x, reduction='sum')

    # KL Divergence
    # Appendix B of Kingma and Welling gives an analytical solution for the KL divergence when
    # 1. The prior is the standard normal distribution, i.e. p_{\theta}(z) = N(z; 0, I)
    # 2. The approximate posterior distribution q_{\phi}(z|x) is Gaussian.

    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_divergence = -0.5 * torch.sum(1 + log_variance - mu.pow(2) - log_variance.exp())

    return reconstruction_loss + kl_divergence

def main() -> None:
    set_seeds()
    print_cuda_configuration()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Configuration ===

    # Path to the directory where downloaded data is stored.
    data_path = './data'

    # Training batch size.
    batch_size = 100

    # Number of passes over the training data.
    num_epochs = 5

    # Learning rate for the optimizer.
    learning_rate = 1e-3

    # === Data ===
    train_loader, test_loader = get_mnist_data(data_path, batch_size)
    image_size = 784  # 28 * 28 pixels.

    vae = VariationalAutoencoder(image_size, 150).to(device)
    vae.train()
    vae.to(device)

    optimizer = optim.Adam(vae.parameters(), lr=1e-3)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}")
        for images, _labels in train_loader:
            images = images.to(device)

            # Forward pass
            outputs, mu, sigma = vae(images)
            loss = loss_function(images, outputs, mu, sigma)
            loss.backward()

            optimizer.zero_grad()
            optimizer.step()

    print("Training complete.")


if __name__ == "__main__":
    main()