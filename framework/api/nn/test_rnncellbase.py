#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test nn.RNNCellBase
"""
from apibase import APIBase, compare
import paddle
import pytest
import numpy as np


class TestRNNCellBase(APIBase):
    """
    test
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32, np.float64]
        # self.debug = True


obj = paddle.nn.RNNCellBase()


@pytest.mark.api_nn_RNNCellBase_vartype
def test_rnncellbase_base():
    """
    dtype=float32
    """
    x = paddle.randn((4, 16))
    shape = (4, 16)
    dtype = paddle.float32

    res = paddle.zeros((4, 4, 16), dtype=paddle.float32)
    paddle_res = obj.get_initial_states(batch_ref=x, shape=shape, dtype=dtype)

    compare(res.numpy(), paddle_res.numpy())


@pytest.mark.api_nn_RNNCellBase_parameters
def test_rnncellbase1():
    """
    input=4D
    """
    x = paddle.randn((6, 10, 3, 22))
    shape = (4, 16)
    dtype = paddle.float32

    res = paddle.zeros((6, 4, 16), dtype=paddle.float32)
    paddle_res = obj.get_initial_states(batch_ref=x, shape=shape, dtype=dtype)

    compare(res.numpy(), paddle_res.numpy())


@pytest.mark.api_nn_RNNCellBase_parameters
def test_rnncellbase2():
    """
    shape=3D
    """
    x = paddle.randn((4, 16))
    shape = (4, 16, 22)
    dtype = paddle.float32

    res = paddle.zeros((4, 4, 16, 22), dtype=paddle.float32)
    paddle_res = obj.get_initial_states(batch_ref=x, shape=shape, dtype=dtype)

    compare(res.numpy(), paddle_res.numpy())


@pytest.mark.api_nn_RNNCellBase_parameters
def test_rnncellbase3():
    """
    init_volume=1
    """
    x = paddle.randn((4, 16))
    shape = (4, 16)
    dtype = paddle.float32
    init_volume = 1

    res = paddle.ones((4, 4, 16), dtype=paddle.float32)
    paddle_res = obj.get_initial_states(batch_ref=x, shape=shape, dtype=dtype, init_value=init_volume)

    compare(res.numpy(), paddle_res.numpy())


@pytest.mark.api_nn_RNNCellBase_parameters
def test_rnncellbase4():
    """
    batch_dim_idx=1
    """
    x = paddle.randn((4, 16))
    shape = (4, 16)
    dtype = paddle.float32
    batch_dim_idx = 1

    res = paddle.zeros((16, 4, 16), dtype=paddle.float32)
    paddle_res = obj.get_initial_states(batch_ref=x, shape=shape, dtype=dtype, batch_dim_idx=batch_dim_idx)

    compare(res.numpy(), paddle_res.numpy())


@pytest.mark.api_nn_RNNCellBase_vartype
def test_rnncellbase5():
    """
    dtype=float64
    """
    x = paddle.randn((4, 16))
    shape = (4, 16)
    dtype = paddle.float64

    res = paddle.zeros((4, 4, 16), dtype=paddle.float64)
    paddle_res = obj.get_initial_states(batch_ref=x, shape=shape, dtype=dtype)

    compare(res.numpy(), paddle_res.numpy())


@pytest.mark.api_nn_RNNCellBase_parameters
def test_rnncellbase6():
    """
    exception:batch_dim_idx invalid
    """
    x = paddle.randn((4, 16))
    shape = (4, 16)
    dtype = paddle.float32
    batch_dim_idx = 2

    try:
        obj.get_initial_states(batch_ref=x, shape=shape, dtype=dtype, batch_dim_idx=batch_dim_idx)
    except Exception as e:
        if "[operator < fill_constant_batch_size_like > error]" in e.args[0]:
            pass
        else:
            raise Exception


@pytest.mark.api_nn_RNNCellBase_parameters
def test_rnncellbase7():
    """
    exception:shape not implemented
    """
    x = paddle.randn((4, 16))
    dtype = paddle.float32

    try:
        obj.get_initial_states(batch_ref=x, dtype=dtype)
    except NotImplementedError as e:
        if "Please add implementaion for `state_shape` in the used cell." in e.args[0]:
            pass
        else:
            raise Exception


@pytest.mark.api_nn_RNNCellBase_parameters
def test_rnncellbase8():
    """
    exception:dtype not implemented
    """
    x = paddle.randn((4, 16))
    shape = (4, 16)

    try:
        obj.get_initial_states(batch_ref=x, shape=shape)
    except NotImplementedError as e:
        if "Please add implementaion for `state_dtype` in the used cell." in e.args[0]:
            pass
        else:
            raise Exception


@pytest.mark.api_nn_RNNCellBase_parameters
def test_rnncellbase9():
    """
    init_volume=-2
    """
    x = paddle.randn((4, 16))
    shape = (4, 16)
    dtype = paddle.float32
    init_volume = -2

    res = paddle.ones((4, 4, 16), dtype=paddle.float32) * -2
    paddle_res = obj.get_initial_states(batch_ref=x, shape=shape, dtype=dtype, init_value=init_volume)

    compare(res.numpy(), paddle_res.numpy())


@pytest.mark.api_nn_RNNCellBase_parameters
def test_rnncellbase10():
    """
    exception:wrong shape
    """
    x = paddle.randn((4, 16))
    shape = (-1, -1)
    dtype = paddle.float32

    try:
        obj.get_initial_states(batch_ref=x, shape=shape, dtype=dtype)
    except Exception as e:
        if "[operator < fill_constant_batch_size_like > error]" in e.args[0]:
            pass
        else:
            raise Exception


@pytest.mark.api_nn_RNNCellBase_parameters
def test_rnncellbas11():
    """
    init_volume=1e22
    """
    x = paddle.randn((4, 16))
    shape = (4, 16)
    dtype = paddle.float32
    init_volume = 1e22

    res = paddle.ones((4, 4, 16), dtype=paddle.float32) * 1e22
    paddle_res = obj.get_initial_states(batch_ref=x, shape=shape, dtype=dtype, init_value=init_volume)

    compare(res.numpy(), paddle_res.numpy())
