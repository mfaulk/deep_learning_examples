import unittest

import torch
from torch import nn

from neural_networks.symmetric_autoencoder import SymmetricAutoencoder


class TestAutoencoder(unittest.TestCase):
    def setUp(self) -> None:
        self.layer_sizes = [784, 256, 64, 16]
        self.autoencoder = SymmetricAutoencoder(self.layer_sizes)

    def test_num_encoder_layers(self) -> None:
        """
        The encoder should contain a linear layer between each pair of sizes in the layer sizes list.
        Each linear layer should be followed by a ReLU activation function.
        """
        layer_sizes = [784, 256, 64, 16]
        expected_num_linear_layers = len(layer_sizes) - 1
        expected_num_relu_layers = len(layer_sizes) - 1
        expected_len = expected_num_linear_layers + expected_num_relu_layers
        autoencoder = SymmetricAutoencoder(layer_sizes)
        self.assertEqual(len(autoencoder.encoder), expected_len)

    def test_encoder_layer_types(self) -> None:
        """
        Test if the encoder layers are of the correct types.
        """
        layer_sizes = [784, 256, 64, 16]
        autoencoder = SymmetricAutoencoder(layer_sizes)
        self.assertIsInstance(autoencoder.encoder[0], nn.Linear)
        self.assertIsInstance(autoencoder.encoder[1], nn.ReLU)
        self.assertIsInstance(autoencoder.encoder[2], nn.Linear)
        self.assertIsInstance(autoencoder.encoder[3], nn.ReLU)
        self.assertIsInstance(autoencoder.encoder[4], nn.Linear)

    def test_encoder_linear_layer_sizes(self) -> None:
        """
        Test if the encoder linear layers have the correct sizes.
        """
        layer_sizes = [784, 256, 64, 16]
        autoencoder = SymmetricAutoencoder(layer_sizes)
        self.assertEqual(autoencoder.encoder[0].in_features, 784)
        self.assertEqual(autoencoder.encoder[0].out_features, 256)
        self.assertEqual(autoencoder.encoder[2].in_features, 256)
        self.assertEqual(autoencoder.encoder[2].out_features, 64)
        self.assertEqual(autoencoder.encoder[4].in_features, 64)
        self.assertEqual(autoencoder.encoder[4].out_features, 16)

    def test_num_decoder_layers(self) -> None:
        """
        The decoder should contain a linear layer between each pair of sizes in the layer sizes list.
        Each linear layer except the last should be followed by a ReLU activation function.
        The last linear layer should be followed by a Tanh activation function.
        """

        layer_sizes = [784, 256, 64, 16]
        expected_num_linear_layers = len(layer_sizes) - 1
        expected_num_relu_layers = len(layer_sizes) - 2
        expected_len = expected_num_linear_layers + expected_num_relu_layers + 1
        autoencoder = SymmetricAutoencoder(layer_sizes)
        self.assertEqual(len(autoencoder.decoder), expected_len)

    def test_decoder_layer_types(self) -> None:
        """
        Test if the decoder layers are of the correct types.
        """
        layer_sizes = [784, 256, 64, 16]
        autoencoder = SymmetricAutoencoder(layer_sizes)
        self.assertIsInstance(autoencoder.decoder[0], nn.Linear)
        self.assertIsInstance(autoencoder.decoder[1], nn.ReLU)
        self.assertIsInstance(autoencoder.decoder[2], nn.Linear)
        self.assertIsInstance(autoencoder.decoder[3], nn.ReLU)
        self.assertIsInstance(autoencoder.decoder[4], nn.Linear)
        self.assertIsInstance(autoencoder.decoder[5], nn.Tanh)

    def test_decoder_linear_layer_sizes(self) -> None:
        """
        Test if the decoder linear layers have the correct sizes.
        """
        layer_sizes = [784, 256, 64, 16]
        autoencoder = SymmetricAutoencoder(layer_sizes)
        self.assertEqual(autoencoder.decoder[0].in_features, 16)
        self.assertEqual(autoencoder.decoder[0].out_features, 64)
        self.assertEqual(autoencoder.decoder[2].in_features, 64)
        self.assertEqual(autoencoder.decoder[2].out_features, 256)
        self.assertEqual(autoencoder.decoder[4].in_features, 256)
        self.assertEqual(autoencoder.decoder[4].out_features, 784)

    def test_forward_pass_output_shapes(self) -> None:
        """
        The forward pass should return an output tensor with the same shape as the input tensor,
        and a latent code tensor with the same shape as the code size.
        """
        layer_sizes = [784, 256, 64, 16]
        # Code size is the last layer size.
        code_size = layer_sizes[-1]
        autoencoder = SymmetricAutoencoder(layer_sizes)
        x = torch.randn(1, 784)
        output, latent_code = autoencoder.forward(x)

        # The output should have the same shape as the input.
        self.assertEqual(output.shape, x.shape)

        # The latent code should have the same shape as the code size.
        self.assertEqual(latent_code.shape, (1, code_size))


if __name__ == "__main__":
    unittest.main()
