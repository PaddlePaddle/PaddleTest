#!/usr/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python

"""
test_randint_like
"""

from apibase import APIBase
import paddle
import pytest
import numpy as np


def cal_randint_like(x, dtype="float32", low=0, high=None):
    """
    calculate api
    """
    x = x.astype(dtype)
    xp = paddle.to_tensor(x)
    dynamic_res = paddle.randint_like(xp, low=low, high=high)

    paddle.enable_static()
    main_program, startup_program = paddle.static.Program(), paddle.static.Program()
    with paddle.utils.unique_name.guard():
        with paddle.static.program_guard(main_program=main_program, startup_program=startup_program):
            data0 = paddle.static.data(name="s0", shape=x.shape, dtype=dtype)
            feed = {"s0": x}
            out = paddle.randint_like(data0, low=low, high=high)

            exe = paddle.static.Executor()
            exe.run(startup_program)
            static_res = exe.run(main_program, feed=feed, fetch_list=[out])
    paddle.disable_static()
    assert dynamic_res.numpy().shape == static_res[0].shape
    return dynamic_res.numpy()


@pytest.mark.api_nn_randint_like_parameters
def test_randint_like0():
    """
    base
    """
    x = np.zeros((4,))
    res = cal_randint_like(x, low=-4, high=4)
    assert res.shape == (4,)


@pytest.mark.api_nn_randint_like_parameters
def test_randint_like1():
    """
    default
    """
    x = np.zeros((4, 5))
    res = cal_randint_like(x, low=-4, high=4)
    assert res.shape == (4, 5)


@pytest.mark.api_nn_randint_like_parameters
def test_randint_like2():
    """
    x: 3-d tensor
    """
    x = np.zeros((2, 4, 5))
    res = cal_randint_like(x, low=-4, high=4)
    assert res.shape == (2, 4, 5)


@pytest.mark.api_nn_randint_like_parameters
def test_randint_like3():
    """
    x: 4-d tensor
    """
    x = np.zeros((2, 2, 4, 5))
    res = cal_randint_like(x, low=-4, high=4)
    assert res.shape == (2, 2, 4, 5)


@pytest.mark.api_nn_randint_like_parameters
def test_randint_like4():
    """
    high=None
    """
    x = np.zeros((4, 5))
    res = cal_randint_like(x, low=4)
    assert res.shape == (4, 5)
