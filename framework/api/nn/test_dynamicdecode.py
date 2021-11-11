#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test paddle.nn.dynamic_decode
"""

from apibase import compare
import paddle, random
import pytest
import numpy as np
from paddle.nn import BeamSearchDecoder, dynamic_decode
from paddle.nn import GRUCell, Linear, Embedding, LSTMCell
from paddle.nn import TransformerDecoderLayer, TransformerDecoder
import paddle.fluid as fluid
import paddle.fluid.layers as layers
from paddle.fluid.layers import GreedyEmbeddingHelper, SampleEmbeddingHelper

np.random.seed(2)
random.seed(2)
paddle.seed(2)


class ModelGRUCell(paddle.nn.Layer):
    """
    GRUCell model
    """

    def __init__(self):
        """
        initialize
        """
        super(ModelGRUCell, self).__init__()
        self.trg_embeder = Embedding(100, 32)
        self.output_layer = Linear(32, 32)
        self.decoder_cell = GRUCell(input_size=32, hidden_size=32)
        self.decoder = BeamSearchDecoder(
            self.decoder_cell,
            start_token=0,
            end_token=1,
            beam_size=4,
            embedding_fn=self.trg_embeder,
            output_fn=self.output_layer,
        )

    def forward(self):
        """
        forward
        """
        encoder_output = paddle.ones((4, 8, 32), dtype=paddle.get_default_dtype())
        outputs = dynamic_decode(
            decoder=self.decoder, inits=self.decoder_cell.get_initial_states(encoder_output), max_step_num=10
        )
        return outputs[0]


class ModelGRUCell1(paddle.nn.Layer):
    """
    GRUCell model1
    """

    def __init__(self):
        """
        initialize
        """
        super(ModelGRUCell1, self).__init__()
        self.trg_embeder = Embedding(100, 32)
        self.output_layer = Linear(32, 32)
        self.decoder_cell = GRUCell(input_size=32, hidden_size=32)
        self.decoder = BeamSearchDecoder(
            self.decoder_cell,
            start_token=0,
            end_token=1,
            beam_size=4,
            embedding_fn=self.trg_embeder,
            output_fn=self.output_layer,
        )

    def forward(self):
        """
        forward
        """
        encoder_output = paddle.ones((4, 8, 32), dtype=paddle.get_default_dtype())
        outputs = dynamic_decode(
            decoder=self.decoder,
            inits=self.decoder_cell.get_initial_states(encoder_output),
            output_time_major=True,
            max_step_num=10,
        )
        return outputs[0]


class ModelGRUCell2(paddle.nn.Layer):
    """
    GRUCell model2
    """

    def __init__(self):
        """
        initialize
        """
        super(ModelGRUCell2, self).__init__()
        self.trg_embeder = Embedding(100, 32)
        self.output_layer = Linear(32, 32)
        self.decoder_cell = GRUCell(input_size=32, hidden_size=32)
        self.decoder = BeamSearchDecoder(
            self.decoder_cell,
            start_token=0,
            end_token=1,
            beam_size=4,
            embedding_fn=self.trg_embeder,
            output_fn=self.output_layer,
        )

    def forward(self):
        """
        forward
        """
        encoder_output = paddle.ones((4, 8, 32), dtype=paddle.get_default_dtype())
        outputs = dynamic_decode(
            decoder=self.decoder,
            inits=self.decoder_cell.get_initial_states(encoder_output),
            is_test=True,
            max_step_num=10,
        )
        return outputs[0]


class ModelGRUCell3(paddle.nn.Layer):
    """
    GRUCell model3
    """

    def __init__(self):
        """
        initialize
        """
        super(ModelGRUCell3, self).__init__()
        self.trg_embeder = Embedding(100, 32)
        self.output_layer = Linear(32, 32)
        self.decoder_cell = GRUCell(input_size=32, hidden_size=32)
        self.decoder = BeamSearchDecoder(
            self.decoder_cell,
            start_token=0,
            end_token=1,
            beam_size=4,
            embedding_fn=self.trg_embeder,
            output_fn=self.output_layer,
        )

    def forward(self):
        """
        forward
        """
        encoder_output = paddle.ones((4, 8, 32), dtype=paddle.get_default_dtype())
        outputs = dynamic_decode(
            decoder=self.decoder,
            inits=self.decoder_cell.get_initial_states(encoder_output),
            impute_finished=True,
            max_step_num=10,
        )
        return outputs[0]


class ModelGRUCell4(paddle.nn.Layer):
    """
    GRUCell model4
    """

    def __init__(self):
        """
        initialize
        """
        super(ModelGRUCell4, self).__init__()
        self.trg_embeder = Embedding(100, 32)
        self.output_layer = Linear(32, 32)
        self.decoder_cell = GRUCell(input_size=32, hidden_size=32)
        self.decoder = BeamSearchDecoder(
            self.decoder_cell,
            start_token=0,
            end_token=1,
            beam_size=4,
            embedding_fn=self.trg_embeder,
            output_fn=self.output_layer,
        )

    def forward(self):
        """
        forward
        """
        encoder_output = paddle.ones((4, 8, 32), dtype=paddle.get_default_dtype())
        outputs = dynamic_decode(
            decoder=self.decoder,
            inits=self.decoder_cell.get_initial_states(encoder_output),
            return_length=True,
            max_step_num=10,
        )
        return outputs[2]


class ModelGRUCell5(paddle.nn.Layer):
    """
    GRUCell model5  greedy
    """

    def __init__(self):
        """
        initialize
        """
        super(ModelGRUCell5, self).__init__()
        self.trg_embeder = Embedding(100, 8)
        self.output_layer = Linear(8, 8)
        start_token = fluid.layers.fill_constant(shape=[2], dtype="int64", value=0)
        self.helper = GreedyEmbeddingHelper(self.trg_embeder, start_tokens=start_token, end_token=1)
        self.decoder_cell = GRUCell(input_size=8, hidden_size=8)
        self.decoder = layers.BasicDecoder(self.decoder_cell, self.helper, output_fn=self.output_layer)

    def forward(self):
        """
        forward
        """
        encoder_output = paddle.ones((2, 4, 8), dtype=paddle.get_default_dtype())
        outputs = dynamic_decode(
            decoder=self.decoder, inits=self.decoder_cell.get_initial_states(encoder_output), max_step_num=5
        )
        return outputs[0][0]


class ModelGRUCell6(paddle.nn.Layer):
    """
    GRUCell model6 sampling
    """

    def __init__(self):
        """
        initialize
        """
        super(ModelGRUCell6, self).__init__()
        self.trg_embeder = Embedding(100, 8)
        self.output_layer = Linear(8, 8)
        start_token = fluid.layers.fill_constant(shape=[2], dtype="int64", value=0)
        self.helper = SampleEmbeddingHelper(self.trg_embeder, start_tokens=start_token, end_token=1, seed=2)
        self.decoder_cell = GRUCell(input_size=8, hidden_size=8)
        self.decoder = layers.BasicDecoder(self.decoder_cell, self.helper, output_fn=self.output_layer)

    def forward(self):
        """
        forward
        """
        encoder_output = paddle.ones((2, 4, 8), dtype=paddle.get_default_dtype())
        outputs = dynamic_decode(
            decoder=self.decoder, inits=self.decoder_cell.get_initial_states(encoder_output), max_step_num=5
        )
        return outputs[0][0]


class ModelLSTMCell(paddle.nn.Layer):
    """
    LSTMCell model
    """

    def __init__(self):
        """
        initialize
        """
        super(ModelLSTMCell, self).__init__()
        self.trg_embeder = Embedding(100, 32)
        self.output_layer = Linear(32, 32)
        self.decoder_cell = LSTMCell(input_size=32, hidden_size=32)
        self.decoder = BeamSearchDecoder(
            self.decoder_cell,
            start_token=0,
            end_token=1,
            beam_size=4,
            embedding_fn=self.trg_embeder,
            output_fn=self.output_layer,
        )

    def forward(self):
        """
        forward
        """
        encoder_output = paddle.ones((4, 8, 32), dtype=paddle.get_default_dtype())
        outputs = dynamic_decode(
            decoder=self.decoder, inits=self.decoder_cell.get_initial_states(encoder_output), max_step_num=10
        )
        return outputs[0]


class ModelLSTMCell1(paddle.nn.Layer):
    """
    LSTMCell model1
    """

    def __init__(self):
        """
        initialize
        """
        super(ModelLSTMCell1, self).__init__()
        self.trg_embeder = Embedding(100, 16)
        self.output_layer = Linear(16, 16)
        self.decoder_cell = LSTMCell(input_size=16, hidden_size=16)
        self.decoder = BeamSearchDecoder(
            self.decoder_cell,
            start_token=0,
            end_token=1,
            beam_size=4,
            embedding_fn=self.trg_embeder,
            output_fn=self.output_layer,
        )

    def forward(self):
        """
        forward
        """
        encoder_output = paddle.ones((4, 4, 16), dtype=paddle.get_default_dtype())
        outputs = dynamic_decode(
            decoder=self.decoder, inits=self.decoder_cell.get_initial_states(encoder_output), max_step_num=10
        )
        return outputs[0]


class ModelLSTMCell2(paddle.nn.Layer):
    """
    LSTMCell model2
    """

    def __init__(self):
        """
        initialize
        """
        super(ModelLSTMCell2, self).__init__()
        self.trg_embeder = Embedding(100, 32)
        self.output_layer = Linear(32, 32)
        self.decoder_cell = LSTMCell(input_size=32, hidden_size=32)
        self.decoder = BeamSearchDecoder(
            self.decoder_cell,
            start_token=0,
            end_token=1,
            beam_size=4,
            embedding_fn=self.trg_embeder,
            output_fn=self.output_layer,
        )

    def forward(self):
        """
        forward
        """
        encoder_output = paddle.ones((4, 8, 32), dtype=paddle.get_default_dtype())
        outputs = dynamic_decode(
            decoder=self.decoder, inits=self.decoder_cell.get_initial_states(encoder_output), max_step_num=5
        )
        return outputs[0]


@pytest.mark.api_nn_dynamic_decode_parameters
def test_dynamic_decode0():
    """
    GRUCell
    """
    # paddle.seed(33)
    m = ModelGRUCell()
    a = paddle.load("model/model_grucell")
    m.set_state_dict(a)
    res = [
        [
            [23, 23, 23, 23],
            [9, 23, 9, 9],
            [9, 9, 9, 9],
            [9, 9, 9, 9],
            [9, 9, 9, 9],
            [9, 9, 9, 9],
            [9, 9, 9, 9],
            [9, 9, 9, 9],
            [9, 9, 9, 9],
            [9, 9, 9, 9],
            [9, 9, 23, 27],
        ],
        [
            [23, 23, 23, 23],
            [9, 23, 9, 9],
            [9, 9, 9, 9],
            [9, 9, 9, 9],
            [9, 9, 9, 9],
            [9, 9, 9, 9],
            [9, 9, 9, 9],
            [9, 9, 9, 9],
            [9, 9, 9, 9],
            [9, 9, 9, 9],
            [9, 9, 23, 27],
        ],
        [
            [23, 23, 23, 23],
            [9, 23, 9, 9],
            [9, 9, 9, 9],
            [9, 9, 9, 9],
            [9, 9, 9, 9],
            [9, 9, 9, 9],
            [9, 9, 9, 9],
            [9, 9, 9, 9],
            [9, 9, 9, 9],
            [9, 9, 9, 9],
            [9, 9, 23, 27],
        ],
        [
            [23, 23, 23, 23],
            [9, 23, 9, 9],
            [9, 9, 9, 9],
            [9, 9, 9, 9],
            [9, 9, 9, 9],
            [9, 9, 9, 9],
            [9, 9, 9, 9],
            [9, 9, 9, 9],
            [9, 9, 9, 9],
            [9, 9, 9, 9],
            [9, 9, 23, 27],
        ],
    ]
    compare(m().numpy(), res)


@pytest.mark.api_nn_dynamic_decode_parameters
def test_dynamic_decode1():
    """
    change the decoder cell to LSTMCell
    """
    m = ModelLSTMCell()
    a = paddle.load("model/model_lstmcell")
    m.set_state_dict(a)
    res = [
        [
            [4, 4, 22, 4],
            [4, 4, 4, 4],
            [30, 20, 20, 30],
            [30, 30, 30, 30],
            [30, 30, 30, 30],
            [30, 30, 30, 30],
            [30, 30, 30, 30],
            [30, 30, 30, 30],
            [30, 30, 30, 30],
            [30, 30, 30, 30],
            [30, 30, 30, 20],
        ],
        [
            [4, 4, 22, 4],
            [4, 4, 4, 4],
            [30, 20, 20, 30],
            [30, 30, 30, 30],
            [30, 30, 30, 30],
            [30, 30, 30, 30],
            [30, 30, 30, 30],
            [30, 30, 30, 30],
            [30, 30, 30, 30],
            [30, 30, 30, 30],
            [30, 30, 30, 20],
        ],
        [
            [4, 4, 22, 4],
            [4, 4, 4, 4],
            [30, 20, 20, 30],
            [30, 30, 30, 30],
            [30, 30, 30, 30],
            [30, 30, 30, 30],
            [30, 30, 30, 30],
            [30, 30, 30, 30],
            [30, 30, 30, 30],
            [30, 30, 30, 30],
            [30, 30, 30, 20],
        ],
        [
            [4, 4, 22, 4],
            [4, 4, 4, 4],
            [30, 20, 20, 30],
            [30, 30, 30, 30],
            [30, 30, 30, 30],
            [30, 30, 30, 30],
            [30, 30, 30, 30],
            [30, 30, 30, 30],
            [30, 30, 30, 30],
            [30, 30, 30, 30],
            [30, 30, 30, 20],
        ],
    ]
    compare(m().numpy(), res)


@pytest.mark.api_nn_dynamic_decode_parameters
def test_dynamic_decode2():
    """
    change the input size
    """
    m = ModelLSTMCell1()
    a = paddle.load("model/model_lstmcell1")
    m.set_state_dict(a)
    res = [
        [
            [4, 4, 4, 4],
            [4, 4, 4, 4],
            [4, 4, 4, 4],
            [4, 4, 4, 4],
            [4, 4, 4, 4],
            [4, 4, 4, 4],
            [4, 4, 4, 4],
            [4, 4, 4, 4],
            [4, 4, 4, 4],
            [4, 4, 9, 9],
            [4, 9, 9, 4],
        ],
        [
            [4, 4, 4, 4],
            [4, 4, 4, 4],
            [4, 4, 4, 4],
            [4, 4, 4, 4],
            [4, 4, 4, 4],
            [4, 4, 4, 4],
            [4, 4, 4, 4],
            [4, 4, 4, 4],
            [4, 4, 4, 4],
            [4, 4, 9, 9],
            [4, 9, 9, 4],
        ],
        [
            [4, 4, 4, 4],
            [4, 4, 4, 4],
            [4, 4, 4, 4],
            [4, 4, 4, 4],
            [4, 4, 4, 4],
            [4, 4, 4, 4],
            [4, 4, 4, 4],
            [4, 4, 4, 4],
            [4, 4, 4, 4],
            [4, 4, 9, 9],
            [4, 9, 9, 4],
        ],
        [
            [4, 4, 4, 4],
            [4, 4, 4, 4],
            [4, 4, 4, 4],
            [4, 4, 4, 4],
            [4, 4, 4, 4],
            [4, 4, 4, 4],
            [4, 4, 4, 4],
            [4, 4, 4, 4],
            [4, 4, 4, 4],
            [4, 4, 9, 9],
            [4, 9, 9, 4],
        ],
    ]
    compare(m().numpy(), res)


@pytest.mark.api_nn_dynamic_decode_parameters
def test_dynamic_decode3():
    """
    change the max_step_num
    """
    m = ModelLSTMCell2()
    a = paddle.load("model/model_lstmcell2")
    m.set_state_dict(a)
    res = [
        [[4, 4, 22, 4], [4, 4, 4, 4], [30, 20, 20, 30], [30, 30, 30, 30], [30, 30, 30, 30], [30, 30, 30, 20]],
        [[4, 4, 22, 4], [4, 4, 4, 4], [30, 20, 20, 30], [30, 30, 30, 30], [30, 30, 30, 30], [30, 30, 30, 20]],
        [[4, 4, 22, 4], [4, 4, 4, 4], [30, 20, 20, 30], [30, 30, 30, 30], [30, 30, 30, 30], [30, 30, 30, 20]],
        [[4, 4, 22, 4], [4, 4, 4, 4], [30, 20, 20, 30], [30, 30, 30, 30], [30, 30, 30, 30], [30, 30, 30, 20]],
    ]
    compare(m().numpy(), res)


@pytest.mark.api_nn_dynamic_decode_parameters
def test_dynamic_decode4():
    """
    set the output_time_major True
    """
    m = ModelGRUCell1()
    a = paddle.load("model/model_grucell1")
    m.set_state_dict(a)
    res = [
        [[23, 23, 23, 23], [23, 23, 23, 23], [23, 23, 23, 23], [23, 23, 23, 23]],
        [[9, 23, 9, 9], [9, 23, 9, 9], [9, 23, 9, 9], [9, 23, 9, 9]],
        [[9, 9, 9, 9], [9, 9, 9, 9], [9, 9, 9, 9], [9, 9, 9, 9]],
        [[9, 9, 9, 9], [9, 9, 9, 9], [9, 9, 9, 9], [9, 9, 9, 9]],
        [[9, 9, 9, 9], [9, 9, 9, 9], [9, 9, 9, 9], [9, 9, 9, 9]],
        [[9, 9, 9, 9], [9, 9, 9, 9], [9, 9, 9, 9], [9, 9, 9, 9]],
        [[9, 9, 9, 9], [9, 9, 9, 9], [9, 9, 9, 9], [9, 9, 9, 9]],
        [[9, 9, 9, 9], [9, 9, 9, 9], [9, 9, 9, 9], [9, 9, 9, 9]],
        [[9, 9, 9, 9], [9, 9, 9, 9], [9, 9, 9, 9], [9, 9, 9, 9]],
        [[9, 9, 9, 9], [9, 9, 9, 9], [9, 9, 9, 9], [9, 9, 9, 9]],
        [[9, 9, 23, 27], [9, 9, 23, 27], [9, 9, 23, 27], [9, 9, 23, 27]],
    ]
    compare(m().numpy(), res)


@pytest.mark.api_nn_dynamic_decode_parameters
def test_dynamic_decode5():
    """
    set the is_test True
    """
    m = ModelGRUCell2()
    a = paddle.load("model/model_grucell2")
    m.set_state_dict(a)
    res = [
        [
            [23, 23, 23, 23],
            [9, 23, 9, 9],
            [9, 9, 9, 9],
            [9, 9, 9, 9],
            [9, 9, 9, 9],
            [9, 9, 9, 9],
            [9, 9, 9, 9],
            [9, 9, 9, 9],
            [9, 9, 9, 9],
            [9, 9, 9, 9],
            [9, 9, 23, 27],
        ],
        [
            [23, 23, 23, 23],
            [9, 23, 9, 9],
            [9, 9, 9, 9],
            [9, 9, 9, 9],
            [9, 9, 9, 9],
            [9, 9, 9, 9],
            [9, 9, 9, 9],
            [9, 9, 9, 9],
            [9, 9, 9, 9],
            [9, 9, 9, 9],
            [9, 9, 23, 27],
        ],
        [
            [23, 23, 23, 23],
            [9, 23, 9, 9],
            [9, 9, 9, 9],
            [9, 9, 9, 9],
            [9, 9, 9, 9],
            [9, 9, 9, 9],
            [9, 9, 9, 9],
            [9, 9, 9, 9],
            [9, 9, 9, 9],
            [9, 9, 9, 9],
            [9, 9, 23, 27],
        ],
        [
            [23, 23, 23, 23],
            [9, 23, 9, 9],
            [9, 9, 9, 9],
            [9, 9, 9, 9],
            [9, 9, 9, 9],
            [9, 9, 9, 9],
            [9, 9, 9, 9],
            [9, 9, 9, 9],
            [9, 9, 9, 9],
            [9, 9, 9, 9],
            [9, 9, 23, 27],
        ],
    ]

    compare(m().numpy(), res)


@pytest.mark.api_nn_dynamic_decode_parameters
def test_dynamic_decode6():
    """
    set the impute_finished True
    """
    m = ModelGRUCell3()
    a = paddle.load("model/model_grucell3")
    m.set_state_dict(a)
    res = [
        [
            [23, 23, 23, 23],
            [9, 23, 9, 9],
            [9, 9, 9, 9],
            [9, 9, 9, 9],
            [9, 9, 9, 9],
            [9, 9, 9, 9],
            [9, 9, 9, 9],
            [9, 9, 9, 9],
            [9, 9, 9, 9],
            [9, 9, 9, 9],
            [9, 9, 23, 27],
        ],
        [
            [23, 23, 23, 23],
            [9, 23, 9, 9],
            [9, 9, 9, 9],
            [9, 9, 9, 9],
            [9, 9, 9, 9],
            [9, 9, 9, 9],
            [9, 9, 9, 9],
            [9, 9, 9, 9],
            [9, 9, 9, 9],
            [9, 9, 9, 9],
            [9, 9, 23, 27],
        ],
        [
            [23, 23, 23, 23],
            [9, 23, 9, 9],
            [9, 9, 9, 9],
            [9, 9, 9, 9],
            [9, 9, 9, 9],
            [9, 9, 9, 9],
            [9, 9, 9, 9],
            [9, 9, 9, 9],
            [9, 9, 9, 9],
            [9, 9, 9, 9],
            [9, 9, 23, 27],
        ],
        [
            [23, 23, 23, 23],
            [9, 23, 9, 9],
            [9, 9, 9, 9],
            [9, 9, 9, 9],
            [9, 9, 9, 9],
            [9, 9, 9, 9],
            [9, 9, 9, 9],
            [9, 9, 9, 9],
            [9, 9, 9, 9],
            [9, 9, 9, 9],
            [9, 9, 23, 27],
        ],
    ]

    compare(m().numpy(), res)


@pytest.mark.api_nn_dynamic_decode_parameters
def test_dynamic_decode7():
    """
    set the return_length True
    """
    m = ModelGRUCell4()
    a = paddle.load("model/model_grucell4")
    m.set_state_dict(a)
    res = [[11, 11, 11, 11], [11, 11, 11, 11], [11, 11, 11, 11], [11, 11, 11, 11]]
    compare(m().numpy(), res)


@pytest.mark.api_nn_dynamic_decode_parameters
def test_dynamic_decode8():
    """
    change the type of decoder to greedy decoder
    """
    m = ModelGRUCell5()
    a = paddle.load("model/model_grucell5")
    m.set_state_dict(a)
    res = [
        [
            [0.11869079, 0.1519133, 0.1153042, 0.10923727, -0.10902043, -0.08880261, 0.21914443, -0.02771666],
            [0.2698136, 0.16878146, 0.14045529, 0.14249605, -0.16314404, -0.20360196, 0.27845696, 0.05379843],
            [0.33944094, 0.18998903, 0.15686867, 0.15775543, -0.19299456, -0.25513914, 0.31023985, 0.08508033],
            [0.2853232, 0.29707664, 0.20960297, 0.18791078, -0.21615289, -0.21297555, 0.38392943, -0.01484178],
            [0.35605448, 0.26608512, 0.19605184, 0.1790646, -0.22442842, -0.27070168, 0.37305287, 0.04462929],
            [0.38736784, 0.25219566, 0.1892334, 0.17557402, -0.23131351, -0.2940567, 0.36857957, 0.07194422],
        ],
        [
            [0.11869079, 0.1519133, 0.1153042, 0.10923727, -0.10902043, -0.08880261, 0.21914443, -0.02771666],
            [0.2698136, 0.16878146, 0.14045529, 0.14249605, -0.16314404, -0.20360196, 0.27845696, 0.05379843],
            [0.33944094, 0.18998903, 0.15686867, 0.15775543, -0.19299456, -0.25513914, 0.31023985, 0.08508033],
            [0.2853232, 0.29707664, 0.20960297, 0.18791078, -0.21615289, -0.21297555, 0.38392943, -0.01484178],
            [0.35605448, 0.26608512, 0.19605184, 0.1790646, -0.22442842, -0.27070168, 0.37305287, 0.04462929],
            [0.38736784, 0.25219566, 0.1892334, 0.17557402, -0.23131351, -0.2940567, 0.36857957, 0.07194422],
        ],
    ]

    compare(m().numpy(), res)


@pytest.mark.api_nn_dynamic_decode_parameters
def test_dynamic_decode9():
    """
    change the type of decoder to sampling decoder
    """
    m = ModelGRUCell6()
    a = paddle.load("model/model_grucell6")
    m.set_state_dict(a)
    res = [
        [
            [0.11869079, 0.1519133, 0.1153042, 0.10923727, -0.10902043, -0.08880261, 0.21914443, -0.02771666],
            [0.1479352, 0.15751627, 0.1285398, 0.14293213, -0.1752117, -0.09633583, 0.3730936, -0.03824619],
            [0.16957042, 0.17692488, 0.12954153, 0.1616616, -0.20020072, -0.10258074, 0.43376634, -0.05128284],
            [0.18431416, 0.19203474, 0.12747352, 0.17047578, -0.21184054, -0.10847434, 0.46083933, -0.06100671],
            [0.19431548, 0.2019853, 0.12548834, 0.1746127, -0.2187091, -0.11357243, 0.47472844, -0.06734832],
            [0.2010567, 0.20840105, 0.12431937, 0.17666325, -0.2232733, -0.11757688, 0.48273546, -0.07123636],
        ],
        [
            [0.11869079, 0.1519133, 0.1153042, 0.10923727, -0.10902043, -0.08880261, 0.21914443, -0.02771666],
            [0.2698136, 0.16878146, 0.14045529, 0.14249605, -0.16314404, -0.20360196, 0.27845696, 0.05379843],
            [0.33944094, 0.18998903, 0.15686867, 0.15775543, -0.19299456, -0.25513914, 0.31023985, 0.08508033],
            [0.37261504, 0.20561421, 0.16585062, 0.1639725, -0.2107406, -0.27936527, 0.32857555, 0.09616168],
            [0.3893159, 0.21575813, 0.17029321, 0.16630685, -0.22173324, -0.29145893, 0.33959395, 0.09949352],
            [0.3982478, 0.22209346, 0.17235991, 0.16710962, -0.22865176, -0.29787856, 0.34638044, 0.10008325],
        ],
    ]

    compare(m().numpy(), res)


@pytest.mark.api_nn_dynamic_decode_exception
def test_dynamic_decode10():
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
def test_dynamic_decode11():
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
def test_dynamic_decode12():
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
