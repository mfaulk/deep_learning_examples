# Deep Learning Examples

Deep Learning Examples with PyTorch.

## Examples

1. [MNIST PCA](src/examples/mnist_pca.py): Principal Component Analysis of MNIST dataset,
2. [MNIST Autoencoder](src/examples/mnist_autoencoder.py): Autoencoder,
3. [MNIST Autoencoder Cross Validation](src/examples/mnist_autoencoder_cross_validation.py): Model selection via cross-validation,
2. [MNIST Convolutional NN](src/examples/mnist_classifier.py): Convolutional neural network for the MNIST dataset,

## Tools

This project uses several tools for code quality, testing, and documentation:

- **[Poetry](https://python-poetry.org/)**: Manages dependencies and virtual environments for consistent builds and setup
- **[PyTorch](https://pytorch.org/)**: The core deep learning library
- **[mypy](http://mypy-lang.org/)**: Static type checking
- **[Ruff](https://beta.ruff.rs/)**: Fast and comprehensive linter
- **[pytest](https://pytest.org/)**: Flexible framework for unit testing
- **[Coverage.py](https://coverage.readthedocs.io/)**: Measures code coverage
- **[Jupyter Notebook](https://jupyter.org/)**: Interactive development for experimenting and prototyping

## Setup

### CUDA: GPU Support

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

### Install Dependencies
Install Poetry (if you haven't already) and install the project dependencies:

```bash
curl -sSL https://install.python-poetry.org | python3 -
poetry install
``` 

## Usage

To run an example, e.g. the MNIST Autoencoder, use a command like:

```bash
poetry run python -m examples.mnist_autoencoder
```

## Jupyter Notebooks
To start Jupyter and run a notebook, use the following command:

```bash
poetry run jupyter notebook --notebook-dir ./notebooks
```

## Testing
To run the tests with coverage (and, optionally, with mypy) use:

```bash
poetry run pytest
poetry run pytest --mypy
```

## MyPy
Static type checking with MyPy:
```bash
poetry run mypy src tests
```  

## Ruff

Checks the codebase (minus notebooks) for linting issues. Add `--fix` to automatically fix some issues.

```bash
poetry run ruff check src tests
poetry run ruff check src tests --fix
```

Format the codebase:

```bash
poetry run ruff format src tests
```
