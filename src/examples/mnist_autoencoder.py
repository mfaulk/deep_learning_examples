from typing import List

import matplotlib.pyplot as plt
import torch
from torch import Tensor, optim
from torch import nn
from torchinfo import summary
from torchvision import transforms as transforms

from datasets.mnist import mnist
from neural_networks.evaluating import evaluate_autoencoder
from neural_networks.symmetric_autoencoder import SymmetricAutoencoder
from neural_networks.training import train_autoencoder
from src.utils.cuda import print_cuda_configuration
from src.utils.seeds import set_seeds


def display_reconstructions(
    original: Tensor, reconstructed: Tensor, num_display: int = 5
) -> None:
    """
    Display the original and reconstructed images.
    :param original: Original images.
    :param reconstructed: Reconstructed images.
    :param num_display: Number of original-reconstructed image pairs to display.
    :return:
    """
    fig, axes = plt.subplots(
        nrows=2, ncols=num_display, sharex=True, sharey=True, figsize=(20, 4)
    )
    for images, row in zip([original, reconstructed], axes):
        for img, ax in zip(images, row):
            ax.imshow(img.view(28, 28).detach().numpy(), cmap="gray")
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)


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
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
            torch.flatten,
        ]
    )
    train_loader, test_loader = mnist(data_path, batch_size, transform)
    image_size = 784  # 28 * 28 pixels.

    # === Training ===
    criterion: nn.MSELoss = nn.MSELoss()
    layers = [image_size, 2500, 150]
    model: SymmetricAutoencoder = SymmetricAutoencoder(layers).cuda()
    summary(model, input_size=(batch_size, image_size))
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    per_batch_loss: List[float] = train_autoencoder(
        model, device, train_loader, optimizer, num_epochs
    )

    # === Testing ===

    avg_test_loss = evaluate_autoencoder(model, device, test_loader, criterion)
    print(f"\nAverage Testing Loss: {avg_test_loss:.4f}")

    # === Visualizations ===

    test_data_iter = iter(test_loader)
    test_images, _labels = next(test_data_iter)
    test_images = test_images.cuda()

    (reconstructed, _code) = model.forward(test_images)
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
    main()
