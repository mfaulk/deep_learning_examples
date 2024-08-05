import itertools
from typing import List, Tuple


def generate_configurations(min_depth: int, max_depth: int, widths: List[int]) -> List[List[int]]:
    """
    Generate all possible neural network configurations given a range of depths and widths.

    Parameters:
    min_depth (int): Minimum number of layers in the neural network.
    max_depth (int): Maximum number of layers in the neural network.
    widths (List[int]): List of integers specifying possible layer widths.

    Returns:
    List[List[int]]: A list containing all possible configurations. Each configuration is represented as a list of integers, where each integer specifies the number of neurons in a layer.
    """

    # min_depth <= max_depth
    if min_depth > max_depth:
        raise ValueError("min_depth must be less than or equal to max_depth.")

    all_configurations: List[Tuple[int, ...]] = []

    for depth in range(min_depth, max_depth + 1):
        layer_ranges = [widths] * depth
        configurations = itertools.product(*layer_ranges)
        all_configurations.extend(configurations)

    # Convert tuples to lists
    return [list(config) for config in all_configurations]
