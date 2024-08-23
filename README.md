# Deep Learning Examples

A Deep Learning Quickstart using PyTorch.

## Examples

1. [MNIST Autoencoder](src/mnist_autoencoder): A simple autoencoder network for the MNIST dataset.
2. MNIST VAE: A variational autoencoder network for the MNIST dataset.

## Tools

This project template configures tools for code quality, testing, and documentation:

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

### Poetry: Dependency Management

If you haven't installed Poetry yet, you can do so by following the instructions below:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

### Install Dependencies

```bash
poetry install
``` 

## Usage

To train the MNIST autoencoder, run the following command:

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
MyPy is a static type checker for Python that helps developers ensure their code is type-safe.
By analyzing type annotations in your code, MyPy can catch potential type errors before runtime
```bash
poetry run mypy src tests
```  

## Ruff

Checks the codebase (minus notebooks) for linting issues. Add `--fix` to automatically fix some issues.

```bash
poetry run ruff check src tests
poetry run ruff check src tests --fix
```

