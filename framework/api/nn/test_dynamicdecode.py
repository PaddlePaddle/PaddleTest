#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test paddle.nn.dynamic_decode
"""

import random
import paddle
from apibase import compare
import pytest
import numpy as np
from paddle.nn import BeamSearchDecoder, dynamic_decode
from paddle.nn import GRUCell, Linear, Embedding, LSTMCell
from paddle.nn import TransformerDecoderLayer, TransformerDecoder

np.random.seed(2)
random.seed(2)
paddle.seed(2)


class ModelGRUCell4(paddle.nn.Layer):
    """
    GRUCell model
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
            decoder=self.decoder, inits=self.decoder_cell.get_initial_states(encoder_output), max_step_num=10
        )
        return outputs[0]


class ModelGRUCell5(paddle.nn.Layer):
    """
    GRUCell model1
    """

    def __init__(self):
        """
        initialize
        """
        super(ModelGRUCell5, self).__init__()
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


class ModelGRUCell6(paddle.nn.Layer):
    """
    GRUCell model2
    """

    def __init__(self):
        """
        initialize
        """
        super(ModelGRUCell6, self).__init__()
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


class ModelGRUCell7(paddle.nn.Layer):
    """
    GRUCell model3
    """

    def __init__(self):
        """
        initialize
        """
        super(ModelGRUCell7, self).__init__()
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


class ModelGRUCell8(paddle.nn.Layer):
    """
    GRUCell model4
    """

    def __init__(self):
        """
        initialize
        """
        super(ModelGRUCell8, self).__init__()
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


class ModelLSTMCell1(paddle.nn.Layer):
    """
    LSTMCell model
    """

    def __init__(self):
        """
        initialize
        """
        super(ModelLSTMCell1, self).__init__()
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


class ModelLSTMCell2(paddle.nn.Layer):
    """
    LSTMCell model1
    """

    def __init__(self):
        """
        initialize
        """
        super(ModelLSTMCell2, self).__init__()
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


class ModelLSTMCell3(paddle.nn.Layer):
    """
    LSTMCell model2
    """

    def __init__(self):
        """
        initialize
        """
        super(ModelLSTMCell3, self).__init__()
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
    m = ModelGRUCell4()
    a = paddle.load("model/model_grucell4")
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
    m = ModelLSTMCell1()
    a = paddle.load("model/model_lstmcell1")
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
    m = ModelLSTMCell2()
    a = paddle.load("model/model_lstmcell2")
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
    m = ModelLSTMCell3()
    a = paddle.load("model/model_lstmcell3")
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
    m = ModelGRUCell5()
    a = paddle.load("model/model_grucell5")
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
    m = ModelGRUCell6()
    a = paddle.load("model/model_grucell6")
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
    m = ModelGRUCell7()
    a = paddle.load("model/model_grucell7")
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
    m = ModelGRUCell8()
    a = paddle.load("model/model_grucell8")
    m.set_state_dict(a)
    res = [[11, 11, 11, 11], [11, 11, 11, 11], [11, 11, 11, 11], [11, 11, 11, 11]]
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


@pytest.mark.skip(reason="RD代码异常改变，此Case会报错，暂时跳过")
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


@pytest.mark.skip(reason="RD代码异常改变，此Case会报错，暂时跳过")
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
