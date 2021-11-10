#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test paddle.nn.dynamic_decode
"""

from apibase import APIBase
from apibase import compare
import paddle
import pytest
import numpy as np
from paddle.nn import BeamSearchDecoder, dynamic_decode
from paddle.nn import GRUCell, Linear, Embedding, LSTMCell
from paddle.nn import TransformerDecoderLayer, TransformerDecoder
import paddle.fluid as fluid
import paddle.fluid.layers as layers
from paddle.fluid.layers import GreedyEmbeddingHelper, SampleEmbeddingHelper


class TestDynamicDecode(APIBase):
    """
    test
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32, np.float64, np.int64, np.int32, bool]
        # self.debug = True
        # self.static = True
        # enable check grad
        # self.enable_backward = True


# obj = TestDynamicDecode(paddle.nn.dynamic_decode)


@pytest.mark.api_nn_dynamic_decode_vartype
def test_dynamic_decode_base():
    """
    base
    """
    paddle.seed(33)
    trg_embeder = Embedding(100, 32)
    output_layer = Linear(32, 32)
    decoder_cell = GRUCell(input_size=32, hidden_size=32)
    decoder = BeamSearchDecoder(
        decoder_cell, start_token=0, end_token=1, beam_size=4, embedding_fn=trg_embeder, output_fn=output_layer
    )
    encoder_output = paddle.ones((4, 8, 32), dtype=paddle.get_default_dtype())
    outputs = dynamic_decode(decoder=decoder, inits=decoder_cell.get_initial_states(encoder_output), max_step_num=10)
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
    compare(np.array(outputs[0]), res)


@pytest.mark.api_nn_dynamic_decode_parameters
def test_dynamic_decode1():
    """
    change the type of decoder_cell
    """
    paddle.seed(33)
    trg_embeder = Embedding(100, 32)
    output_layer = Linear(32, 32)
    decoder_cell = LSTMCell(input_size=32, hidden_size=32)
    decoder = BeamSearchDecoder(
        decoder_cell, start_token=0, end_token=1, beam_size=4, embedding_fn=trg_embeder, output_fn=output_layer
    )
    encoder_output = paddle.ones((4, 8, 32), dtype=paddle.get_default_dtype())
    outputs = dynamic_decode(decoder=decoder, inits=decoder_cell.get_initial_states(encoder_output), max_step_num=10)
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
    ]
    compare(np.array(outputs[0]), res)


@pytest.mark.api_nn_dynamic_decode_parameters
def test_dynamic_decode2():
    """
    change the input size
    """
    paddle.seed(33)
    trg_embeder = Embedding(100, 16)
    output_layer = Linear(16, 16)
    decoder_cell = GRUCell(input_size=16, hidden_size=16)
    decoder = BeamSearchDecoder(
        decoder_cell, start_token=0, end_token=1, beam_size=4, embedding_fn=trg_embeder, output_fn=output_layer
    )
    encoder_output = paddle.ones((4, 4, 16), dtype=paddle.get_default_dtype())
    outputs = dynamic_decode(decoder=decoder, inits=decoder_cell.get_initial_states(encoder_output), max_step_num=10)
    res = [
        [
            [10, 10, 10, 10],
            [3, 3, 3, 12],
            [12, 12, 12, 3],
            [3, 3, 3, 12],
            [3, 12, 3, 3],
            [12, 3, 3, 3],
            [3, 3, 3, 3],
            [3, 3, 3, 3],
            [3, 3, 3, 3],
            [3, 3, 3, 3],
            [3, 3, 3, 3],
        ],
        [
            [10, 10, 10, 10],
            [3, 3, 3, 12],
            [12, 12, 12, 3],
            [3, 3, 3, 12],
            [3, 12, 3, 3],
            [12, 3, 3, 3],
            [3, 3, 3, 3],
            [3, 3, 3, 3],
            [3, 3, 3, 3],
            [3, 3, 3, 3],
            [3, 3, 3, 3],
        ],
        [
            [10, 10, 10, 10],
            [3, 3, 3, 12],
            [12, 12, 12, 3],
            [3, 3, 3, 12],
            [3, 12, 3, 3],
            [12, 3, 3, 3],
            [3, 3, 3, 3],
            [3, 3, 3, 3],
            [3, 3, 3, 3],
            [3, 3, 3, 3],
            [3, 3, 3, 3],
        ],
        [
            [10, 10, 10, 10],
            [3, 3, 3, 12],
            [12, 12, 12, 3],
            [3, 3, 3, 12],
            [3, 12, 3, 3],
            [12, 3, 3, 3],
            [3, 3, 3, 3],
            [3, 3, 3, 3],
            [3, 3, 3, 3],
            [3, 3, 3, 3],
            [3, 3, 3, 3],
        ],
    ]
    compare(np.array(outputs[0]), res)


@pytest.mark.api_nn_dynamic_decode_parameters
def test_dynamic_decode3():
    """
    change the max_step_num
    """
    paddle.seed(33)
    trg_embeder = Embedding(100, 32)
    output_layer = Linear(32, 32)
    decoder_cell = LSTMCell(input_size=32, hidden_size=32)
    decoder = BeamSearchDecoder(
        decoder_cell, start_token=0, end_token=1, beam_size=4, embedding_fn=trg_embeder, output_fn=output_layer
    )
    encoder_output = paddle.ones((4, 8, 32), dtype=paddle.get_default_dtype())
    outputs = dynamic_decode(decoder=decoder, inits=decoder_cell.get_initial_states(encoder_output), max_step_num=5)
    res = [
        [[14, 14, 14, 14], [2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 10], [2, 10, 2, 2], [2, 2, 10, 2]],
        [[14, 14, 14, 14], [2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 10], [2, 10, 2, 2], [2, 2, 10, 2]],
        [[14, 14, 14, 14], [2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 10], [2, 10, 2, 2], [2, 2, 10, 2]],
        [[14, 14, 14, 14], [2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 10], [2, 10, 2, 2], [2, 2, 10, 2]],
    ]
    compare(np.array(outputs[0]), res)


@pytest.mark.api_nn_dynamic_decode_parameters
def test_dynamic_decode4():
    """
    set the output_time_major True
    """
    paddle.seed(33)
    trg_embeder = Embedding(100, 32)
    output_layer = Linear(32, 32)
    decoder_cell = LSTMCell(input_size=32, hidden_size=32)
    decoder = BeamSearchDecoder(
        decoder_cell, start_token=0, end_token=1, beam_size=4, embedding_fn=trg_embeder, output_fn=output_layer
    )
    encoder_output = paddle.ones((4, 8, 32), dtype=paddle.get_default_dtype())
    outputs = dynamic_decode(
        decoder=decoder, inits=decoder_cell.get_initial_states(encoder_output), output_time_major=True, max_step_num=5
    )
    res = [
        [[14, 14, 14, 14], [14, 14, 14, 14], [14, 14, 14, 14], [14, 14, 14, 14]],
        [[2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2]],
        [[2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2]],
        [[2, 2, 2, 10], [2, 2, 2, 10], [2, 2, 2, 10], [2, 2, 2, 10]],
        [[2, 10, 2, 2], [2, 10, 2, 2], [2, 10, 2, 2], [2, 10, 2, 2]],
        [[2, 2, 10, 2], [2, 2, 10, 2], [2, 2, 10, 2], [2, 2, 10, 2]],
    ]
    compare(np.array(outputs[0]), res)


@pytest.mark.api_nn_dynamic_decode_parameters
def test_dynamic_decode5():
    """
    set the is_test True
    """
    paddle.seed(33)
    trg_embeder = Embedding(100, 32)
    output_layer = Linear(32, 32)
    decoder_cell = LSTMCell(input_size=32, hidden_size=32)
    decoder = BeamSearchDecoder(
        decoder_cell, start_token=0, end_token=1, beam_size=4, embedding_fn=trg_embeder, output_fn=output_layer
    )
    encoder_output = paddle.ones((4, 8, 32), dtype=paddle.get_default_dtype())
    outputs = dynamic_decode(
        decoder=decoder, inits=decoder_cell.get_initial_states(encoder_output), is_test=True, max_step_num=5
    )
    res = [
        [[14, 14, 14, 14], [2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 10], [2, 10, 2, 2], [2, 2, 10, 2]],
        [[14, 14, 14, 14], [2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 10], [2, 10, 2, 2], [2, 2, 10, 2]],
        [[14, 14, 14, 14], [2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 10], [2, 10, 2, 2], [2, 2, 10, 2]],
        [[14, 14, 14, 14], [2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 10], [2, 10, 2, 2], [2, 2, 10, 2]],
    ]
    compare(np.array(outputs[0]), res)


@pytest.mark.api_nn_dynamic_decode_parameters
def test_dynamic_decode6():
    """
    set the return_length True
    """
    paddle.seed(33)
    trg_embeder = Embedding(100, 32)
    output_layer = Linear(32, 32)
    decoder_cell = LSTMCell(input_size=32, hidden_size=32)
    decoder = BeamSearchDecoder(
        decoder_cell, start_token=0, end_token=1, beam_size=4, embedding_fn=trg_embeder, output_fn=output_layer
    )
    encoder_output = paddle.ones((4, 8, 32), dtype=paddle.get_default_dtype())
    outputs = dynamic_decode(
        decoder=decoder, inits=decoder_cell.get_initial_states(encoder_output), return_length=True, max_step_num=5
    )
    res = [[6, 6, 6, 6], [6, 6, 6, 6], [6, 6, 6, 6], [6, 6, 6, 6]]
    compare(np.array(outputs[2]), res)


@pytest.mark.api_nn_dynamic_decode_parameters
def test_dynamic_decode7():
    """
    change the type of decoder to greedy decoder
    """
    paddle.seed(33)
    trg_embeder = Embedding(100, 8)
    output_layer = Linear(8, 8)
    start_token = fluid.layers.fill_constant(shape=[2], dtype="int64", value=0)
    helper = GreedyEmbeddingHelper(trg_embeder, start_tokens=start_token, end_token=1)
    decoder_cell = GRUCell(input_size=8, hidden_size=8)
    decoder = layers.BasicDecoder(decoder_cell, helper, output_fn=output_layer)
    encoder_output = paddle.ones((2, 4, 8), dtype=paddle.get_default_dtype())
    outputs = dynamic_decode(decoder=decoder, inits=decoder_cell.get_initial_states(encoder_output), max_step_num=5)
    res = [
        [
            [-0.17127818, -0.01234272, -0.21454653, 0.11479405, 0.06952278, 0.10348236, 0.19047457, 0.06276155],
            [-0.30858770, -0.03620958, -0.23224403, 0.21163929, -0.01110586, 0.18284672, 0.31641164, 0.04559596],
            [-0.36297917, -0.04205710, -0.23866612, 0.28525308, -0.04433357, 0.21380942, 0.37001812, 0.05405118],
            [-0.38365650, -0.04405222, -0.24023992, 0.33436719, -0.06027451, 0.22934601, 0.39576593, 0.06609423],
            [-0.39140514, -0.04537192, -0.23976383, 0.36590716, -0.06949674, 0.23900747, 0.40986329, 0.07574011],
            [-0.39429501, -0.04653242, -0.23862694, 0.38603976, -0.07555486, 0.24570876, 0.41833925, 0.08242066],
        ],
        [
            [-0.17127818, -0.01234272, -0.21454653, 0.11479405, 0.06952278, 0.10348236, 0.19047457, 0.06276155],
            [-0.30858770, -0.03620958, -0.23224403, 0.21163929, -0.01110586, 0.18284672, 0.31641164, 0.04559596],
            [-0.36297917, -0.04205710, -0.23866612, 0.28525308, -0.04433357, 0.21380942, 0.37001812, 0.05405118],
            [-0.38365650, -0.04405222, -0.24023992, 0.33436719, -0.06027451, 0.22934601, 0.39576593, 0.06609423],
            [-0.39140514, -0.04537192, -0.23976383, 0.36590716, -0.06949674, 0.23900747, 0.40986329, 0.07574011],
            [-0.39429501, -0.04653242, -0.23862694, 0.38603976, -0.07555486, 0.24570876, 0.41833925, 0.08242066],
        ],
    ]
    compare(np.array(outputs[0][0]), res)


@pytest.mark.api_nn_dynamic_decode_parameters
def test_dynamic_decode8():
    """
    change the type of decoder to sampling decoder
    """
    paddle.seed(33)
    trg_embeder = Embedding(100, 8)
    output_layer = Linear(8, 8)
    start_token = fluid.layers.fill_constant(shape=[2], dtype="int64", value=0)
    decoder_cell = GRUCell(input_size=8, hidden_size=8)
    helper = SampleEmbeddingHelper(trg_embeder, start_tokens=start_token, end_token=1, seed=33)
    decoder = layers.BasicDecoder(decoder_cell, helper, output_fn=output_layer)
    encoder_output = paddle.ones((2, 4, 8), dtype=paddle.get_default_dtype())
    outputs = dynamic_decode(decoder=decoder, inits=decoder_cell.get_initial_states(encoder_output), max_step_num=3)
    res = [
        [
            [-0.17127818, -0.01234272, -0.21454653, 0.11479405, 0.06952278, 0.10348236, 0.19047457, 0.06276155],
            [-0.32198220, -0.05770380, -0.27743614, 0.19352689, -0.00347658, 0.21470174, 0.28232136, 0.07605741],
            [-0.39296269, -0.06796025, -0.29694632, 0.25277194, -0.03548478, 0.24359798, 0.32911450, 0.08541774],
            [-0.42532066, -0.06987520, -0.30312890, 0.28966117, -0.05014505, 0.25121012, 0.35385773, 0.09215528],
        ],
        [
            [-0.17127818, -0.01234272, -0.21454653, 0.11479405, 0.06952278, 0.10348236, 0.19047457, 0.06276155],
            [-0.32198220, -0.05770380, -0.27743614, 0.19352689, -0.00347658, 0.21470174, 0.28232136, 0.07605741],
            [-0.39296269, -0.06796025, -0.29694632, 0.25277194, -0.03548478, 0.24359798, 0.32911450, 0.08541774],
            [-0.42532066, -0.06987520, -0.30312890, 0.28966117, -0.05014505, 0.25121012, 0.35385773, 0.09215528],
        ],
    ]
    compare(np.array(outputs[0][0]), res)


@pytest.mark.api_nn_dynamic_decode_exception
def test_dynamic_decode9():
    """
    Decoder type error
    """
    decoder_cell = LSTMCell(input_size=32, hidden_size=32)
    output_layer = TransformerDecoderLayer(32, 2, 128)
    decoder = TransformerDecoder(output_layer, 2)
    encoder_output = paddle.ones((4, 8, 32), dtype=paddle.get_default_dtype())
    try:
        dynamic_decode(decoder=decoder, inits=decoder_cell.get_initial_states(encoder_output), max_step_num=10)
    except Exception as e:
        # print(e)
        if "object has no attribute 'initialize'" in e.args[0]:
            pass
        else:
            raise Exception


@pytest.mark.api_nn_dynamic_decode_exception
def test_dynamic_decode10():
    """
    No parameters passed to inits
    """
    paddle.seed(33)
    trg_embeder = Embedding(100, 32)
    output_layer = Linear(32, 32)
    decoder_cell = GRUCell(input_size=32, hidden_size=32)
    decoder = BeamSearchDecoder(
        decoder_cell, start_token=0, end_token=1, beam_size=4, embedding_fn=trg_embeder, output_fn=output_layer
    )
    try:
        dynamic_decode(decoder=decoder, max_step_num=5)
    except Exception as e:
        # print(e)
        error = "'NoneType' object has no attribute 'dtype'"
        if error in e.args[0]:
            pass
        else:
            raise Exception


@pytest.mark.api_nn_dynamic_decode_exception
def test_dynamic_decode11():
    """
    the size of inits mismatch the size of the decoder
    """
    paddle.seed(33)
    trg_embeder = Embedding(100, 32)
    output_layer = Linear(32, 32)
    decoder_cell = LSTMCell(input_size=32, hidden_size=32)
    decoder = BeamSearchDecoder(
        decoder_cell, start_token=0, end_token=1, beam_size=4, embedding_fn=trg_embeder, output_fn=output_layer
    )
    encoder_output = paddle.ones((4, 8, 32), dtype=paddle.get_default_dtype())
    decoder_initial_states = [
        decoder_cell.get_initial_states(encoder_output, shape=[16]),
        decoder_cell.get_initial_states(encoder_output, shape=[16]),
    ]
    try:
        dynamic_decode(decoder=decoder, inits=decoder_initial_states, max_step_num=5)
    except Exception as e:
        if "[operator < matmul_v2 > error]" in e.args[0]:
            pass
        else:
            raise Exception
