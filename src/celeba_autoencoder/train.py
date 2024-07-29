import time

from matplotlib import pyplot as plt
from torch import nn, optim
from torchinfo import summary

from celeba_autoencoder.autoencoder import Autoencoder
from celeba_autoencoder.load import load_celeba
from utils.cuda import print_cuda_configuration
from utils.seeds import set_seeds


def main():
    # Set seeds for reproducibility
    set_seeds()

    print_cuda_configuration()

    config = {
        # Directory where downloaded data is stored.
        'data_dir': '/home/mfaulk/data/celeba/CelebA/Img/img_align_celeba',

        # Batch size for training
        'batch_size': 100,

        # Number of passes over the training data
        'num_epochs': 3,

        # Learning rate for the optimizer
        'lr': 1e-3,
    }

    train_loader, test_loader = load_celeba(config['data_dir'], config['batch_size'])

    autoencoder: Autoencoder = Autoencoder()
    autoencoder.cuda()
    summary(autoencoder, input_size=(config['batch_size'], 218 * 178 * 3))

    loss_fn: nn.MSELoss = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=config['lr'])

    # Per-batch training loss
    per_batch_loss = []

    # === Training ===
    for epoch in range(config['num_epochs']):
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
        print(f'Epoch [{epoch + 1}/{config["num_epochs"]}], Loss: {loss.item():.4f}, Time: {elapsed_time:.2f} seconds')

    # Testing loss

    # Plot the per-batch training loss.
    plt.figure(1)
    plt.plot(per_batch_loss)
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title('Per-batch Training Loss')

    plt.show()


if __name__ == '__main__':
    main()