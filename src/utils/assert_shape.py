"""
A custom Module that checks whether the input tensor has the expected shape.
"""
from typing import Tuple

from torch import nn, Tensor


class AssertShape(nn.Module):
    def __init__(self, expected_shape: Tuple[int, ...]) -> None:
        """
        Custom module that checks whether the input tensor has the expected shape.

        Args:
            expected_shape (Tuple[int, ...]): The expected shape of the input tensor, excluding the batch size.
        """
        super(AssertShape, self).__init__()
        self.expected_shape = expected_shape

    def forward(self, x: Tensor) -> Tensor:
        """
        Checks the shape of the input tensor.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The same tensor if the shape matches the expected shape.

        Raises:
            ValueError: If the input tensor shape does not match the expected shape.
        """
        if x.shape[1:] != self.expected_shape:
            raise ValueError(f"Expected input with shape [batch_size, {self.expected_shape}], but got {x.shape}")
        return x