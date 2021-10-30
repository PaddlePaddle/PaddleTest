#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test paddle.nn.BeamSearchDecoder
"""

from apibase import APIBase
from apibase import compare
import paddle
import pytest
import numpy as np
from paddle.nn import BeamSearchDecoder, dynamic_decode
from paddle.nn import GRUCell, Linear, Embedding, LSTMCell, SimpleRNNCell, Softmax


class TestBeamSearchDecoder(APIBase):
    """
    test
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32, np.float64, np.uint8, np.int64, np.int32, bool]
        # self.debug = True
        # self.static = True
        # enable check grad
        # self.enable_backward = True


obj = TestBeamSearchDecoder(paddle.nn.BeamSearchDecoder)


@pytest.mark.api_nn_BeamSearchDecoder_vartype
def test_beamsearchdecoder_base():
    """
    base
    """
    paddle.seed(obj.seed)
    trg_embeder = Embedding(100, 32)
    output_layer = Linear(32, 32)
    decoder_cell = GRUCell(input_size=32, hidden_size=32)
    decoder = BeamSearchDecoder(
        decoder_cell, start_token=0, end_token=1, beam_size=4, embedding_fn=trg_embeder, output_fn=output_layer
    )
    encoder_output = paddle.ones((4, 8, 32), dtype=paddle.get_default_dtype())
    outputs = dynamic_decode(decoder=decoder, inits=decoder_cell.get_initial_states(encoder_output), max_step_num=10)
    output = outputs[0]
    res = [
        [
            [9, 3, 3, 28],
            [3, 3, 9, 3],
            [3, 3, 3, 3],
            [3, 3, 3, 3],
            [3, 3, 3, 3],
            [3, 3, 3, 3],
            [3, 3, 3, 3],
            [3, 3, 3, 3],
            [3, 3, 3, 3],
            [3, 3, 3, 3],
            [3, 3, 3, 3],
        ],
        [
            [9, 3, 3, 28],
            [3, 3, 9, 3],
            [3, 3, 3, 3],
            [3, 3, 3, 3],
            [3, 3, 3, 3],
            [3, 3, 3, 3],
            [3, 3, 3, 3],
            [3, 3, 3, 3],
            [3, 3, 3, 3],
            [3, 3, 3, 3],
            [3, 3, 3, 3],
        ],
        [
            [9, 3, 3, 28],
            [3, 3, 9, 3],
            [3, 3, 3, 3],
            [3, 3, 3, 3],
            [3, 3, 3, 3],
            [3, 3, 3, 3],
            [3, 3, 3, 3],
            [3, 3, 3, 3],
            [3, 3, 3, 3],
            [3, 3, 3, 3],
            [3, 3, 3, 3],
        ],
        [
            [9, 3, 3, 28],
            [3, 3, 9, 3],
            [3, 3, 3, 3],
            [3, 3, 3, 3],
            [3, 3, 3, 3],
            [3, 3, 3, 3],
            [3, 3, 3, 3],
            [3, 3, 3, 3],
            [3, 3, 3, 3],
            [3, 3, 3, 3],
            [3, 3, 3, 3],
        ],
    ]
    compare(np.array(output), res)


@pytest.mark.api_nn_LayerNorm_parameters
def test_beamsearchdecoder1():
    """
    change decoder_cell to LSTM cell
    """
    paddle.seed(obj.seed)
    trg_embeder = Embedding(100, 32)
    output_layer = Linear(32, 32)
    decoder_cell = LSTMCell(input_size=32, hidden_size=32)
    decoder = BeamSearchDecoder(
        decoder_cell, start_token=0, end_token=1, beam_size=4, embedding_fn=trg_embeder, output_fn=output_layer
    )
    encoder_output = paddle.ones((4, 8, 32), dtype=paddle.get_default_dtype())
    outputs = dynamic_decode(decoder=decoder, inits=decoder_cell.get_initial_states(encoder_output), max_step_num=10)
    output = outputs[0]
    res = [
        [
            [14, 14, 14, 14],
            [2, 2, 2, 2],
            [2, 2, 2, 2],
            [2, 2, 2, 2],
            [2, 2, 2, 2],
            [2, 2, 2, 2],
            [2, 2, 2, 2],
            [2, 2, 2, 2],
            [2, 2, 10, 2],
            [10, 2, 2, 2],
            [2, 2, 2, 10],
        ],
        [
            [14, 14, 14, 14],
            [2, 2, 2, 2],
            [2, 2, 2, 2],
            [2, 2, 2, 2],
            [2, 2, 2, 2],
            [2, 2, 2, 2],
            [2, 2, 2, 2],
            [2, 2, 2, 2],
            [2, 2, 10, 2],
            [10, 2, 2, 2],
            [2, 2, 2, 10],
        ],
        [
            [14, 14, 14, 14],
            [2, 2, 2, 2],
            [2, 2, 2, 2],
            [2, 2, 2, 2],
            [2, 2, 2, 2],
            [2, 2, 2, 2],
            [2, 2, 2, 2],
            [2, 2, 2, 2],
            [2, 2, 10, 2],
            [10, 2, 2, 2],
            [2, 2, 2, 10],
        ],
        [
            [14, 14, 14, 14],
            [2, 2, 2, 2],
            [2, 2, 2, 2],
            [2, 2, 2, 2],
            [2, 2, 2, 2],
            [2, 2, 2, 2],
            [2, 2, 2, 2],
            [2, 2, 2, 2],
            [2, 2, 10, 2],
            [10, 2, 2, 2],
            [2, 2, 2, 10],
        ],
    ]  # [4,11,4]
    compare(np.array(output), res)


@pytest.mark.api_nn_LayerNorm_parameters
def test_beamsearchdecoder2():
    """
    change decoder_cell to simpleRNNCell
    """
    paddle.seed(obj.seed)
    trg_embeder = Embedding(100, 32)
    output_layer = Linear(32, 32)
    decoder_cell = SimpleRNNCell(input_size=32, hidden_size=32)
    decoder = BeamSearchDecoder(
        decoder_cell, start_token=0, end_token=1, beam_size=4, embedding_fn=trg_embeder, output_fn=output_layer
    )
    encoder_output = paddle.ones((4, 8, 32), dtype=paddle.get_default_dtype())
    outputs = dynamic_decode(decoder=decoder, inits=decoder_cell.get_initial_states(encoder_output), max_step_num=10)
    output = outputs[0]
    res = [
        [
            [16, 16, 16, 16],
            [6, 6, 6, 6],
            [6, 6, 6, 6],
            [15, 6, 15, 15],
            [15, 15, 15, 15],
            [15, 15, 15, 15],
            [6, 15, 15, 6],
            [15, 15, 15, 15],
            [15, 15, 15, 15],
            [15, 15, 15, 15],
            [15, 15, 15, 6],
        ],
        [
            [16, 16, 16, 16],
            [6, 6, 6, 6],
            [6, 6, 6, 6],
            [15, 6, 15, 15],
            [15, 15, 15, 15],
            [15, 15, 15, 15],
            [6, 15, 15, 6],
            [15, 15, 15, 15],
            [15, 15, 15, 15],
            [15, 15, 15, 15],
            [15, 15, 15, 6],
        ],
        [
            [16, 16, 16, 16],
            [6, 6, 6, 6],
            [6, 6, 6, 6],
            [15, 6, 15, 15],
            [15, 15, 15, 15],
            [15, 15, 15, 15],
            [6, 15, 15, 6],
            [15, 15, 15, 15],
            [15, 15, 15, 15],
            [15, 15, 15, 15],
            [15, 15, 15, 6],
        ],
        [
            [16, 16, 16, 16],
            [6, 6, 6, 6],
            [6, 6, 6, 6],
            [15, 6, 15, 15],
            [15, 15, 15, 15],
            [15, 15, 15, 15],
            [6, 15, 15, 6],
            [15, 15, 15, 15],
            [15, 15, 15, 15],
            [15, 15, 15, 15],
            [15, 15, 15, 6],
        ],
    ]
    compare(np.array(output), res)


@pytest.mark.api_nn_LayerNorm_parameters
def test_beamsearchdecoder3():
    """
    change the beam size
    """
    paddle.seed(obj.seed)
    trg_embeder = Embedding(100, 32)
    output_layer = Linear(32, 32)
    decoder_cell = GRUCell(input_size=32, hidden_size=32)
    decoder = BeamSearchDecoder(
        decoder_cell, start_token=0, end_token=1, beam_size=8, embedding_fn=trg_embeder, output_fn=output_layer
    )
    encoder_output = paddle.ones((4, 8, 32), dtype=paddle.get_default_dtype())
    outputs = dynamic_decode(decoder=decoder, inits=decoder_cell.get_initial_states(encoder_output), max_step_num=10)
    output = outputs[0]
    res = [
        [
            [9, 3, 3, 9, 3, 28, 27, 6],
            [3, 3, 9, 27, 3, 3, 3, 3],
            [3, 3, 3, 3, 9, 3, 3, 3],
            [3, 3, 3, 3, 3, 3, 3, 3],
            [3, 3, 3, 3, 3, 3, 3, 3],
            [3, 3, 3, 3, 3, 3, 3, 3],
            [3, 3, 3, 3, 3, 3, 3, 3],
            [3, 3, 3, 3, 3, 3, 3, 3],
            [3, 3, 3, 3, 3, 3, 3, 3],
            [3, 3, 3, 3, 3, 3, 3, 3],
            [3, 3, 3, 3, 3, 3, 3, 3],
        ],
        [
            [9, 3, 3, 9, 3, 28, 27, 6],
            [3, 3, 9, 27, 3, 3, 3, 3],
            [3, 3, 3, 3, 9, 3, 3, 3],
            [3, 3, 3, 3, 3, 3, 3, 3],
            [3, 3, 3, 3, 3, 3, 3, 3],
            [3, 3, 3, 3, 3, 3, 3, 3],
            [3, 3, 3, 3, 3, 3, 3, 3],
            [3, 3, 3, 3, 3, 3, 3, 3],
            [3, 3, 3, 3, 3, 3, 3, 3],
            [3, 3, 3, 3, 3, 3, 3, 3],
            [3, 3, 3, 3, 3, 3, 3, 3],
        ],
        [
            [9, 3, 3, 9, 3, 28, 27, 6],
            [3, 3, 9, 27, 3, 3, 3, 3],
            [3, 3, 3, 3, 9, 3, 3, 3],
            [3, 3, 3, 3, 3, 3, 3, 3],
            [3, 3, 3, 3, 3, 3, 3, 3],
            [3, 3, 3, 3, 3, 3, 3, 3],
            [3, 3, 3, 3, 3, 3, 3, 3],
            [3, 3, 3, 3, 3, 3, 3, 3],
            [3, 3, 3, 3, 3, 3, 3, 3],
            [3, 3, 3, 3, 3, 3, 3, 3],
            [3, 3, 3, 3, 3, 3, 3, 3],
        ],
        [
            [9, 3, 3, 9, 3, 28, 27, 6],
            [3, 3, 9, 27, 3, 3, 3, 3],
            [3, 3, 3, 3, 9, 3, 3, 3],
            [3, 3, 3, 3, 3, 3, 3, 3],
            [3, 3, 3, 3, 3, 3, 3, 3],
            [3, 3, 3, 3, 3, 3, 3, 3],
            [3, 3, 3, 3, 3, 3, 3, 3],
            [3, 3, 3, 3, 3, 3, 3, 3],
            [3, 3, 3, 3, 3, 3, 3, 3],
            [3, 3, 3, 3, 3, 3, 3, 3],
            [3, 3, 3, 3, 3, 3, 3, 3],
        ],
    ]  # [4,11,8]
    compare(np.array(output), res)


@pytest.mark.api_nn_LayerNorm_parameters
def test_beamsearchdecoder4():
    """
    change the size of the input sequence
    """
    paddle.seed(obj.seed)
    trg_embeder = Embedding(100, 16)
    output_layer = Linear(16, 16)
    decoder_cell = GRUCell(input_size=16, hidden_size=16)
    decoder = BeamSearchDecoder(
        decoder_cell, start_token=0, end_token=1, beam_size=4, embedding_fn=trg_embeder, output_fn=output_layer
    )
    encoder_output = paddle.ones((4, 8, 16), dtype=paddle.get_default_dtype())
    outputs = dynamic_decode(decoder=decoder, inits=decoder_cell.get_initial_states(encoder_output), max_step_num=5)
    output = outputs[0]
    res = [
        [[10, 10, 10, 10], [3, 3, 12, 3], [12, 12, 3, 12], [3, 3, 12, 3], [12, 3, 3, 3], [3, 3, 3, 12]],
        [[10, 10, 10, 10], [3, 3, 12, 3], [12, 12, 3, 12], [3, 3, 12, 3], [12, 3, 3, 3], [3, 3, 3, 12]],
        [[10, 10, 10, 10], [3, 3, 12, 3], [12, 12, 3, 12], [3, 3, 12, 3], [12, 3, 3, 3], [3, 3, 3, 12]],
        [[10, 10, 10, 10], [3, 3, 12, 3], [12, 12, 3, 12], [3, 3, 12, 3], [12, 3, 3, 3], [3, 3, 3, 12]],
    ]
    compare(np.array(output), res)


@pytest.mark.api_nn_LayerNorm_parameters
def test_beamsearchdecoder5():
    """
    change the type of the output_layer
    """
    paddle.seed(obj.seed)
    trg_embeder = Embedding(100, 16)
    output_layer = Softmax(-1)
    # output_layer=Linear(16,16)
    decoder_cell = GRUCell(input_size=16, hidden_size=16)

    decoder = BeamSearchDecoder(
        decoder_cell, start_token=0, end_token=1, beam_size=4, embedding_fn=trg_embeder, output_fn=output_layer
    )
    encoder_output = paddle.ones((4, 8, 16), dtype=paddle.get_default_dtype())
    outputs = dynamic_decode(decoder=decoder, inits=decoder_cell.get_initial_states(encoder_output), max_step_num=5)
    output = outputs[0]
    res = [
        [[1, 11, 0, 13], [1, 13, 13, 13], [1, 13, 13, 13], [1, 13, 13, 13], [1, 13, 13, 13], [1, 13, 13, 13]],
        [[1, 11, 0, 13], [1, 13, 13, 13], [1, 13, 13, 13], [1, 13, 13, 13], [1, 13, 13, 13], [1, 13, 13, 13]],
        [[1, 11, 0, 13], [1, 13, 13, 13], [1, 13, 13, 13], [1, 13, 13, 13], [1, 13, 13, 13], [1, 13, 13, 13]],
        [[1, 11, 0, 13], [1, 13, 13, 13], [1, 13, 13, 13], [1, 13, 13, 13], [1, 13, 13, 13], [1, 13, 13, 13]],
    ]
    compare(np.array(output), res)


@pytest.mark.api_nn_LayerNorm_parameters
def test_beamsearchdecoder6():
    """
    error shape
    input_size mismatch hidden_size
    """
    paddle.seed(obj.seed)
    trg_embeder = Embedding(100, 16)
    output_layer = Linear(16, 16)
    decoder_cell = GRUCell(input_size=16, hidden_size=32)
    decoder = BeamSearchDecoder(
        decoder_cell, start_token=0, end_token=1, beam_size=4, embedding_fn=trg_embeder, output_fn=output_layer
    )
    encoder_output = paddle.ones((4, 8, 16), dtype=paddle.get_default_dtype())
    try:
        dynamic_decode(decoder=decoder, inits=decoder_cell.get_initial_states(encoder_output), max_step_num=5)
    except Exception as e:
        print(e)
        if ("[operator < matmul > error]" in e.args[0]) or ("[operator < matmul_v2 > error]" in e.args[0]):
            pass
        else:
            raise Exception
