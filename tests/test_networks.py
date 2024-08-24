import unittest
from typing import List

from model_selection.configuration_space import generate_configurations


class TestGenerateConfigurations(unittest.TestCase):
    def test_generate_configurations_depth_0(self) -> None:
        min_depth = 0
        max_depth = 0
        min_width = 2
        max_width = 4
        widths = list(range(min_width, max_width + 1))
        expected_configurations: List[List[int]] = [[]]  # TODO: should this be []?
        configurations = generate_configurations(min_depth, max_depth, widths)
        self.assertEqual(configurations, expected_configurations)

    def test_generate_configurations_depth_1(self) -> None:
        min_depth = 1
        max_depth = 1
        min_width = 2
        max_width = 4
        widths = list(range(min_width, max_width + 1))
        expected_configurations = [[2], [3], [4]]
        configurations = generate_configurations(min_depth, max_depth, widths)
        self.assertEqual(configurations, expected_configurations)

    # Each configuration should contain widths from min_width to max_width (inclusive).
    def test_generate_configurations_widths(self) -> None:
        min_depth = 2
        max_depth = 2
        min_width = 2
        max_width = 4
        widths = list(range(min_width, max_width + 1))

        configurations = generate_configurations(min_depth, max_depth, widths)
        for config in configurations:
            self.assertTrue(all([min_width <= width <= max_width for width in config]))

    # Each configuration should have depth between min_depth and max_depth (inclusive).
    def test_generate_configurations_depth(self) -> None:
        min_depth = 1
        max_depth = 8
        min_width = 3
        max_width = 9
        widths = list(range(min_width, max_width + 1))

        configurations = generate_configurations(min_depth, max_depth, widths)
        for config in configurations:
            self.assertTrue(min_depth <= len(config) <= max_depth)

    # Should contain all possible configurations.
    def test_generate_configurations_count(self) -> None:
        min_depth = 1
        max_depth = 2
        widths = [10, 20]

        expected_configurations = [[10], [20], [10, 10], [10, 20], [20, 10], [20, 20]]
        configurations = generate_configurations(min_depth, max_depth, widths)
        self.assertEqual(len(configurations), len(expected_configurations))
