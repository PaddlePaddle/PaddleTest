#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_functional_embedding
"""

import paddle
import pytest
import numpy as np


# places
if paddle.device.is_compiled_with_cuda() is True:
    paddle.device.set_device("gpu:0")
    places = [paddle.CPUPlace(), paddle.CUDAPlace(0)]
else:
    paddle.device.set_device("cpu")
    places = [paddle.CPUPlace()]

# types
types = ["int32", "int64"]


def cal_embedding_api(x, weight, place, padding_idx=None):
    """
    calculate dynamic and static result of embedding api
    """
    xp = paddle.to_tensor(x)
    wp = paddle.to_tensor(weight)
    dynamic_res = paddle.nn.functional.embedding(xp, wp, padding_idx=padding_idx)

    paddle.enable_static()
    startup_program, main_program = paddle.static.Program(), paddle.static.Program()
    with paddle.utils.unique_name.guard():
        with paddle.static.program_guard(main_program=main_program, startup_program=startup_program):
            data0 = paddle.static.data(name="s0", shape=x.shape, dtype=x.dtype)
            data1 = paddle.static.data(name="s1", shape=weight.shape, dtype="float64")
            feed = {"s0": x, "s1": weight}
            out = paddle.nn.functional.embedding(data0, data1, padding_idx=padding_idx)
            exe = paddle.static.Executor(place)
            exe.run(startup_program)
            static_res = exe.run(main_program, feed=feed, fetch_list=[out])
    paddle.disable_static()
    assert np.allclose(static_res, dynamic_res.numpy())
    return static_res


def cal_embedding(x, weight, padding_idx=None):
    """
    calculate embedding
    """
    shape = list(x.shape)
    embedding_dim = weight.shape[-1]
    new_shape = shape + [embedding_dim]
    if padding_idx:
        padding_idx = weight.shape[0] + padding_idx if padding_idx < 0 else padding_idx
    res = []
    for i in x.flatten():
        if i == padding_idx:
            res.append([0] * embedding_dim)
        else:
            res.append(weight[i])
    return np.array(res).reshape(new_shape)


@pytest.mark.api_nn_embedding_vartype
def test_embedding_base():
    """
    base
    """
    x = np.arange(4)
    w = np.random.rand(10, 4)
    for place in places:
        for dtype in types:
            x = x.astype(dtype)
            api_res = cal_embedding_api(x, w, place)
            res = cal_embedding(x, w)
            assert np.allclose(api_res, res)


@pytest.mark.api_nn_embedding_parameter
def test_embedding0():
    """
    default
    """
    for place in places:
        x = np.random.randint(2, 7, (3, 4))
        w = np.random.rand(10, 4)
        api_res = cal_embedding_api(x, w, place)
        res = cal_embedding(x, w)
        assert np.allclose(api_res, res)


@pytest.mark.api_nn_embedding_parameter
def test_embedding1():
    """
    padding_idx > 0
    """
    for place in places:
        x = np.random.randint(2, 7, (3, 4))
        w = np.random.rand(10, 4)
        padding_idx = 9
        api_res = cal_embedding_api(x, w, place, padding_idx=padding_idx)
        res = cal_embedding(x, w, padding_idx=padding_idx)
        assert np.allclose(api_res, res)


@pytest.mark.api_nn_embedding_parameter
def test_embedding2():
    """
    padding_idx < 0
    """
    for place in places:
        x = np.random.randint(2, 7, (3, 4))
        w = np.random.rand(10, 4)
        padding_idx = -4
        api_res = cal_embedding_api(x, w, place, padding_idx=padding_idx)
        res = cal_embedding(x, w, padding_idx=padding_idx)
        assert np.allclose(api_res, res)
