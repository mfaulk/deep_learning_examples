# PyTorch Examples

A collection of neural network examples using PyTorch.

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- CUDA
- Poetry (for dependency management)

## CUDA Project Requirements

1. **CUDA-Compatible GPU**: Check the [CUDA-enabled products](https://developer.nvidia.com/cuda-gpus) list to see if
   your GPU
   is
   supported.

1. **NVIDIA Drivers**: Install the latest drivers for your GPU from
   the [NVIDIA website](https://www.nvidia.com/Download/index.aspx).

1. **CUDA Toolkit**: Download and install the [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive). The
   version
   of the CUDA Toolkit should match the version of the cuDNN library.

1. **cuDNN Library**: GPU-accelerated primitives for deep neural networks. Download and install
   the [cuDNN library](https://developer.nvidia.com/cudnn).

### Installing Poetry

If you haven't installed Poetry yet, you can do so by following the instructions below:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

### Clone the Repository

```bash
git clone https://github.com/yourusername/pytorch_examples.git
cd pytorch_examples
```

### Install Dependencies

```bash
poetry install
``` 

## Usage

To train the MNIST autoencoder, run the following command:

```bash
poetry run python src/mnist_autoencoder/train.py
```

## Contributing

Contributions are welcome! Please fork the repository and submit pull requests for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

