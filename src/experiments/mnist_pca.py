import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

from datasets.mnist import get_mnist_data


def main() -> None:
    # Path to the directory where downloaded data is stored.
    data_path = './data'

    # Training batch size.
    batch_size = 100
    mnist_train, _mnist_test = get_mnist_data(data_path, batch_size)
    dataloader = DataLoader(mnist_train, batch_size=batch_size, shuffle=False)

    flat_images = []
    for images, _ in dataloader.dataset:
        images = images.view(images.size(0), -1)  # Flatten the images
        flat_images.append(images)

    data = torch.cat(flat_images, dim=0).numpy()

    # Perform PCA and calculate the cumulative explained variance.
    pca = PCA()
    pca.fit(data)
    explained_variance_ratio = np.cumsum(pca.explained_variance_ratio_)

    # The number of components needed to explain 95% of the variance
    n_components_95 = np.where(explained_variance_ratio >= 0.95)[0][0] + 1
    print(f'Number of components to explain 95% variance: {n_components_95}')

    plt.figure(figsize=(10, 6))
    plt.plot(explained_variance_ratio, linewidth=2)
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Intrinsic Dimensionality of MNIST Dataset')
    plt.grid()
    plt.show()


if __name__ == '__main__':
    main()
