from celeba_autoencoder.load import load_celeba
from utils.cuda import print_cuda_configuration
from utils.seeds import set_seeds


def main():
    # Set seeds for reproducibility
    set_seeds()

    print_cuda_configuration()

    config = {
        # Directory where downloaded data is stored.
        'data_dir': './data',

        # Batch size for training
        'batch_size': 100,

        # Number of passes over the training data
        'num_epochs': 3,

        # Learning rate for the optimizer
        'lr': 1e-3,
    }

    train_loader, test_loader = load_celeba(config['data_dir'], config['batch_size'])

    # Display statistics about training and testing data
    print(f"Number of CelebA training examples: {len(train_loader)}")
    print(f"Number of CelebA testing examples: {len(test_loader)}")

    # autoencoder: Autoencoder = Autoencoder()
    # autoencoder.cuda()
    # summary(autoencoder, input_size=(config['batch_size'], 3 * 64 * 64))

    # loss_fn: nn.MSELoss = nn.MSELoss()
    # optimizer = optim.Adam(autoencoder.parameters(), lr=config['lr'])
    #
    # # Per-batch training loss
    # per_batch_loss = []
    #
    # # === Training ===
    # for epoch in range(config['num_epochs']):
    #     start_time = time.time()  # Start time for the epoch
    #     for img_batch, _labels in train_loader:
    #         img_batch = img_batch.cuda()  # Move batch to GPU
    #
    #         # Forward pass


if __name__ == '__main__':
    main()
