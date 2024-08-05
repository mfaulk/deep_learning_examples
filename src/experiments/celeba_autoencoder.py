import time

from matplotlib import pyplot as plt
from torch import nn, optim, Tensor
from torchinfo import summary

from datasets.celeba import load_celeba
from neural_networks.symmetric_autoencoder import SymmetricAutoencoder
from utils.cuda import print_cuda_configuration
from utils.seeds import set_seeds


def display_reconstructions(original: Tensor, reconstructed: Tensor, num_display: int = 10) -> None:
    """
    Display the original and reconstructed images.
    :param original: Original images.
    :param reconstructed: Reconstructed images.
    :param num_display: Number of original-reconstructed image pairs to display.
    :return:
    """
    fig, axes = plt.subplots(nrows=2, ncols=num_display, sharex=True, sharey=True, figsize=(20, 4))
    for images, row in zip([original, reconstructed], axes):
        for img, ax in zip(images, row):
            ax.imshow(img.view(218, 178, 3).detach().numpy())
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)


def main() -> None:
    # Set seeds for reproducibility
    set_seeds()

    print_cuda_configuration()

    celeba_path = '/home/mfaulk/data/celeba/CelebA/Img/img_align_celeba'

    batch_size = 200

    # Number of passes over the training data
    num_epochs = 3

    # Learning rate for the Adam optimizer.
    learning_rate = 1e-3

    train_loader, test_loader = load_celeba(celeba_path, batch_size)

    autoencoder = SymmetricAutoencoder([218 * 178 * 3, 500, 320])
    autoencoder.cuda()
    summary(autoencoder, input_size=(batch_size, 218 * 178 * 3))

    loss_fn: nn.MSELoss = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)

    # Per-batch training loss
    per_batch_loss = []

    # === Training ===
    for epoch in range(num_epochs):
        start_time = time.time()  # Start time for the epoch
        for img_batch, _labels in train_loader:
            img_batch = img_batch.cuda()  # Move batch to GPU

            # Forward pass
            output, code = autoencoder(img_batch)
            loss = loss_fn(output, img_batch)
            per_batch_loss.append(loss.item())

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        elapsed_time = time.time() - start_time
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Time: {elapsed_time:.2f} seconds')

    # Monitor the per-batch training loss.
    plt.figure(1)
    plt.plot(per_batch_loss)
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title('Per-batch Training Loss')

    # Testing
    test_data_iter = iter(test_loader)
    test_images, _labels = next(test_data_iter)
    test_images = test_images.cuda()

    (reconstructed, _code) = autoencoder.forward(test_images)

    # test_images = test_images.cpu()
    # reconstructed = reconstructed.cpu()
    # display_reconstructions(test_images, reconstructed)

    plt.show()


if __name__ == '__main__':
    main()
