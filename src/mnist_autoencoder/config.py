# MNIST Autoencoder Configuration
class TrainingConfig:
    def __init__(self, batch_size: int, epochs: int, learning_rate: float) -> None:
        # Training batch size.
        self.batch_size: int = batch_size

        # Number of passes over the training data.
        self.epochs: int = epochs

        # Learning rate for the optimizer.
        self.learning_rate: float = learning_rate
