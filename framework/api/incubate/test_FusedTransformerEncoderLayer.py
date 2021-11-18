#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_FusedTransformerEncoderLayer
"""
import sys
import paddle
import paddle.nn.initializer as initializer
from paddle.incubate.nn import FusedTransformerEncoderLayer
import pytest
import numpy as np

sys.path.append("../../utils/")
from interceptor import skip_not_compile_gpu


def cal_dynamic(input_data, mask_data, normalize_before=False, activation="relu"):
    """
    calculate api dynamic result
    """
    # encoder input: [batch_size, src_len, d_model]
    enc_input = paddle.to_tensor(input_data)
    # self attention mask: [batch_size, n_head, src_len, src_len]
    attn_mask = paddle.to_tensor(mask_data)
    encoder_layer = FusedTransformerEncoderLayer(
        4,
        2,
        4,
        dropout_rate=0,
        weight_attr=initializer.Constant(2),
        bias_attr=initializer.Constant(2),
        normalize_before=normalize_before,
        activation=activation,
    )
    enc_output = encoder_layer(enc_input, attn_mask)
    return enc_output.numpy()


def cal_static(input_data, mask_data, normalize_before=False, activation="relu"):
    """
    calculate api static result
    """
    paddle.enable_static()
    main_program, startup_program = paddle.static.Program(), paddle.static.Program()
    with paddle.utils.unique_name.guard():
        with paddle.static.program_guard(main_program=main_program, startup_program=startup_program):
            data0 = paddle.static.data(name="s0", shape=input_data.shape, dtype="float32")
            data1 = paddle.static.data(name="s1", shape=mask_data.shape, dtype="float32")
            feed = {"s0": input_data, "s1": mask_data}
            encoder_layer = FusedTransformerEncoderLayer(
                4,
                2,
                4,
                dropout_rate=0,
                weight_attr=initializer.Constant(2),
                bias_attr=initializer.Constant(2),
                normalize_before=normalize_before,
                activation=activation,
            )

            output = encoder_layer(data0, data1)
            exe = paddle.static.Executor()
            exe.run(startup_program)
            res = exe.run(main_program, feed=feed, fetch_list=[output])
            paddle.disable_static()
            return res


@skip_not_compile_gpu
@pytest.mark.api_nn_FusedTransformerEncoderLayer_parameters
def test_FusedTransformerEncoderLayer0():
    """
    default
    """
    np.random.seed(22)
    enc_input_data = np.random.rand(1, 2, 4).astype("float32")
    enc_attn_mask = np.random.rand(1, 1, 2, 2).astype("float32")
    d_r = cal_dynamic(enc_input_data, enc_attn_mask)
    s_r = cal_static(enc_input_data, enc_attn_mask)
    res = np.array(
        [[[-1.20957255, -0.04592677, -0.30633080, 1.56183779], [-1.00477207, -0.14829074, -0.49726188, 1.65031707]]]
    )
    assert np.allclose(d_r, s_r)
    assert np.allclose(d_r, res)


@skip_not_compile_gpu
@pytest.mark.api_nn_FusedTransformerEncoderLayer_parameters
def test_FusedTransformerEncoderLayer1():
    """
    normalize_before=True
    """
    np.random.seed(22)
    enc_input_data = np.random.rand(1, 2, 4).astype("float32")
    enc_attn_mask = np.random.rand(1, 1, 2, 2).astype("float32")
    d_r = cal_dynamic(enc_input_data, enc_attn_mask, normalize_before=True)
    s_r = cal_static(enc_input_data, enc_attn_mask, normalize_before=True)
    res = np.array(
        [
            [
                [164.20846558, 164.48167419, 164.42053223, 164.85917664],
                [164.16990662, 164.33761597, 164.26928711, 164.68978882],
            ]
        ]
    )
    assert np.allclose(d_r, s_r)
    assert np.allclose(d_r, res)


@skip_not_compile_gpu
@pytest.mark.api_nn_FusedTransformerEncoderLayer_parameters
def test_FusedTransformerEncoderLayer2():
    """
    normalize_before=True
    """
    np.random.seed(22)
    enc_input_data = np.random.rand(1, 2, 4).astype("float32")
    enc_attn_mask = np.random.rand(1, 1, 2, 2).astype("float32")
    d_r = cal_dynamic(enc_input_data, enc_attn_mask, normalize_before=True, activation="gelu")
    s_r = cal_static(enc_input_data, enc_attn_mask, normalize_before=True, activation="gelu")
    res = np.array(
        [
            [
                [163.84446716, 164.11767578, 164.05653381, 164.49517822],
                [163.80578613, 163.97351074, 163.90518188, 164.32568359],
            ]
        ]
    )
    assert np.allclose(d_r, s_r)
    assert np.allclose(d_r, res)
