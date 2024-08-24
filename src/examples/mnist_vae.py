"""
Train a Variational Autoencoder (VAE) on the MNIST dataset.

This example follows the VAE paper by Kingma and Welling, Auto-Encoding Variational Bayes, ICLR, 2014.
"""

import matplotlib.pyplot as plt
import torch
from torch import nn, optim, Tensor
from torchvision import transforms as transforms
from torchinfo import summary

from datasets.mnist import mnist
from examples.mnist_autoencoder import display_reconstructions
from neural_networks.variational_autoencoder import VariationalAutoencoder
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
    :return:
        | ||
        || |_
    """
    # Reconstruction loss.
    ce_loss = nn.CrossEntropyLoss()
    reconstruction_loss: Tensor = ce_loss(x_prime, x)

    # KL Divergence
    # Appendix B of Kingma and Welling gives an analytical solution for the KL divergence when
    # 1. The prior is the standard normal distribution, i.e. p_{\theta}(z) = N(z; 0, I)
    # 2. The approximate posterior distribution q_{\phi}(z|x) is Gaussian.

    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_divergence: Tensor = -0.5 * torch.sum(1 + log_variance - mu.pow(2) - log_variance.exp())

    return reconstruction_loss + kl_divergence


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

    # === Data ===
    transform = transforms.Compose(
        [
            transforms.ToTensor(), # Converts pixel values in the range [0, 255] to [0, 1].
            torch.flatten,
        ]
    )
    train_loader, test_loader = mnist(data_path, batch_size, transform)
    image_size = 784  # 28 * 28 pixels.

    vae = VariationalAutoencoder(image_size, 150).to(device)
    summary(vae, input_size=(batch_size, image_size))
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
            epoch_loss += loss.item()

            # Backward pass and parameter updates
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_train_loss = epoch_loss / float(len(train_loader.dataset))
        print(f"  Average Training Loss: {avg_train_loss:.4f}")

    print("Training complete.")

    # === Testing ===

    test_data_iter = iter(test_loader)
    test_images, _labels = next(test_data_iter)
    test_images = test_images.cuda()

    (reconstructed, _, _) = vae.forward(test_images)
    test_images = test_images.cpu()
    reconstructed = reconstructed.cpu()

    # Compare inputs and reconstructed images.
    display_reconstructions(test_images, reconstructed)

    plt.show()


if __name__ == "__main__":
    main()
