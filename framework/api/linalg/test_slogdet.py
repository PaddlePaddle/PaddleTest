#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_slogdet
"""

import paddle
import pytest
import numpy as np


np.random.seed(22)
paddle.seed(22)

if paddle.is_compiled_with_cuda():
    places = [paddle.CPUPlace(), paddle.CUDAPlace(0)]
else:
    places = [paddle.CPUPlace()]

types = ["float32", "float64", "complex64", "complex128"]


def cal_slogdet_api(x, dtype, place):
    """
    calculate paddle.linalg.slogdet
    """
    x = x.astype(dtype)
    xp = paddle.to_tensor(x, dtype=dtype)
    dynamic_res = paddle.linalg.slogdet(xp)

    paddle.enable_static()
    main_program, startup_program = paddle.static.Program(), paddle.static.Program()
    with paddle.utils.unique_name.guard():
        with paddle.static.program_guard(main_program=main_program, startup_program=startup_program):
            data0 = paddle.static.data(name="s0", shape=x.shape, dtype=dtype)
            feed = {"s0": x}
            out = paddle.linalg.slogdet(data0)

            exe = paddle.static.Executor(place)
            exe.run(startup_program)
            static_res = exe.run(main_program, feed=feed, fetch_list=[out])
    paddle.disable_static()

    assert np.allclose(dynamic_res[0].numpy(), static_res[0])
    assert np.allclose(dynamic_res[1].numpy(), static_res[1])

    return static_res


@pytest.mark.api_linalg_slogdet_vartype
def test_slogdet_base():
    """
    base
    """
    x = np.random.rand(14, 14) * 100
    res = np.linalg.slogdet(x)

    for place in places:
        for dtype in types:
            api_res = cal_slogdet_api(x, dtype, place)
            assert np.allclose(res[0], api_res[0])
            assert np.allclose(res[1], api_res[1])


@pytest.mark.api_linalg_slogdet_parameters
def test_slogdet0():
    """
    default
    """
    x = np.random.rand(4, 4)
    res = np.linalg.slogdet(x)

    for place in places:
        for dtype in types:
            api_res = cal_slogdet_api(x, dtype, place)
            assert np.allclose(res[0], api_res[0])
            assert np.allclose(res[1], api_res[1])


@pytest.mark.api_linalg_slogdet_parameters
def test_slogdet1():
    """
    multi_dim
    """
    x = np.random.rand(3, 4, 4)
    res = np.linalg.slogdet(x)

    for place in places:
        for dtype in types:
            api_res = cal_slogdet_api(x, dtype, place)
            assert np.allclose(res[0], api_res[0])
            assert np.allclose(res[1], api_res[1])
