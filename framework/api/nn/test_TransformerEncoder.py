#!/usr/bin/env python

"""
test TransformerEncoder
"""

import unittest
import numpy as np
import pytest
import paddle
from paddle.nn import MultiHeadAttention, TransformerEncoderLayer, TransformerEncoder
from util import *


class Test(unittest.TestCase):
    """
    test
    """

    @pytest.mark.api_nn_TransformerEncoder_parameters
    def test_encoder(self):
        """
        test_encoder
        """
        [
            batch_size,
            d_model,
            n_head,
            dim_feedforward,
            dropout,
            attn_dropout,
            act_dropout,
            sequence_length,
        ] = generate_basic_params(mode="encoder_layer")

        src = np.random.rand(batch_size, sequence_length, d_model).astype("float32")

        src_mask = np.zeros((batch_size, n_head, sequence_length, sequence_length)).astype("float32")
        src_mask[0][0][0][0] = -np.inf
        with fluid.dygraph.guard(fluid.CPUPlace()):
            encoder_layer = TransformerEncoderLayer(d_model, n_head, dim_feedforward, dropout)
            num_layers = 6
            encoder = TransformerEncoder(encoder_layer, num_layers)
            # src, src_mask
            enc_output = encoder(paddle.to_tensor(src), paddle.to_tensor(src_mask))
            print(enc_output)

    @pytest.mark.api_nn_TransformerEncoder_parameters
    def test(self):
        """
        test
        """
        paddle.disable_static()
        # encoder input: [batch_size, src_len, d_model]
        enc_input = paddle.rand((2, 4, 8))
        # self attention mask: [batch_size, n_head, src_len, src_len]
        attn_mask = paddle.rand((2, 2, 4, 4))
        encoder_layer = TransformerEncoderLayer(8, 2, 16)
        encoder = TransformerEncoder(encoder_layer, 2)
        enc_output = encoder(enc_input, attn_mask)  # [2, 4, 8]
        assert enc_output[0].shape == [4, 8]
        assert enc_output.shape == [2, 4, 8]

        paddle.enable_static()
        # encoder input: [batch_size, src_len, d_model]
        enc_input = paddle.rand((2, 4, 8))
        # self attention mask: [batch_size, n_head, src_len, src_len]
        attn_mask = paddle.rand((2, 2, 4, 4))
        encoder_layer = TransformerEncoderLayer(8, 2, 16)
        encoder = TransformerEncoder(encoder_layer, 2)
        enc_output = encoder(enc_input, attn_mask)  # [2, 4, 8]
        assert enc_output[0].shape == (4, 8)
        assert enc_output.shape == (2, 4, 8)


if __name__ == "__main__":
    unittest.main()
