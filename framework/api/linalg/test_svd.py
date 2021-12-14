#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_svd
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

types = ["float32", "float64"]


def cal_svd_api(x, dtype, place, full_matrices=False):
    """
    calculate paddle.linalg.svd
    """
    x = x.astype(dtype)
    xp = paddle.to_tensor(x, dtype=dtype)
    dynamic_res = paddle.linalg.svd(xp, full_matrices=full_matrices)

    paddle.enable_static()
    main_program, startup_program = paddle.static.Program(), paddle.static.Program()
    with paddle.utils.unique_name.guard():
        with paddle.static.program_guard(main_program=main_program, startup_program=startup_program):
            data0 = paddle.static.data(name="s0", shape=x.shape, dtype=dtype)
            feed = {"s0": x}
            out = paddle.linalg.svd(data0, full_matrices=full_matrices)

            exe = paddle.static.Executor(place)
            exe.run(startup_program)
            static_res = exe.run(main_program, feed=feed, fetch_list=[out])
    paddle.disable_static()

    for i in range(3):
        assert np.allclose(dynamic_res[i].numpy(), static_res[i])

    return static_res


@pytest.mark.api_linalg_svd_vartype
def test_svd_base0():
    """
    base
    """
    x = np.random.rand(2, 3)
    res = np.linalg.svd(x, full_matrices=False)

    for place in places:
        for dtype in types:
            api_res = cal_svd_api(x, dtype, place)
            for i in range(3):
                if i == 0 or i == 2:
                    assert np.allclose(np.abs(res[i]), np.abs(api_res[i]))
                else:
                    assert np.allclose(res[i], api_res[i])


@pytest.mark.api_linalg_svd_parameters
def test_svd0():
    """
    x: 3-d tensor
    """
    x = np.random.rand(3, 2, 3)
    res = np.linalg.svd(x, full_matrices=False)

    for place in places:
        dtype = "float64"
        api_res = cal_svd_api(x, dtype, place)
        for i in range(3):
            if i == 0 or i == 2:
                assert np.allclose(np.abs(res[i]), np.abs(api_res[i]))
            else:
                assert np.allclose(res[i], api_res[i])


@pytest.mark.api_linalg_svd_parameters
def test_svd1():
    """
    x: 4-d tensor
    """
    x = np.random.rand(3, 2, 3, 4)
    res = np.linalg.svd(x, full_matrices=False)

    for place in places:
        dtype = "float64"
        api_res = cal_svd_api(x, dtype, place)
        for i in range(3):
            if i == 0 or i == 2:
                assert np.allclose(np.abs(res[i]), np.abs(api_res[i]))
            else:
                assert np.allclose(res[i], api_res[i])


@pytest.mark.api_linalg_svd_parameters
def test_svd2():
    """
    x: 4-d tensor
    full_matrices = True
    """
    x = np.random.rand(3, 2, 3, 4)
    res = np.linalg.svd(x, full_matrices=True)

    for place in places:
        dtype = "float64"
        api_res = cal_svd_api(x, dtype, place, full_matrices=True)
        for i in range(3):
            if i == 0 or i == 2:
                assert np.allclose(np.abs(res[i]), np.abs(api_res[i]))
            else:
                assert np.allclose(res[i], api_res[i])
