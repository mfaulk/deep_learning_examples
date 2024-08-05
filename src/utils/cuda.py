import torch


def print_cuda_configuration() -> None:
    """
    Display CUDA availability and configuration information.
    :return:
    """
    print(f"Torch version {torch.__version__}")

    if torch.cuda.is_available():
        print('CUDA is available. Using GPU')
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"CUDA device capability: {torch.cuda.get_device_capability()}")
        print(f"CUDA device memory: {torch.cuda.get_device_properties(0).total_memory:_}")

    else:
        print('CUDA is not available. Using CPU')
