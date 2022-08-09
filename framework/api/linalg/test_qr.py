#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_qr
"""

from apibase import randtool
import paddle
import pytest
import numpy as np

if paddle.is_compiled_with_cuda():
    places = [paddle.CPUPlace(), paddle.CUDAPlace(0)]
else:
    places = [paddle.CPUPlace()]
types = ["float32", "float64"]

np.random.seed(33)


def cal_static(A, t, place, mode="reduced"):
    """
    calculate static result
    """
    paddle.enable_static()
    main_program, startup_program = paddle.static.Program(), paddle.static.Program()
    with paddle.utils.unique_name.guard():
        with paddle.static.program_guard(main_program=main_program, startup_program=startup_program):
            data = paddle.static.data(name="data", shape=A.shape, dtype=t)
            out = paddle.linalg.qr(data, mode=mode)
            exe = paddle.static.Executor(place)
            exe.run(startup_program)
            res = exe.run(main_program, feed={"data": A}, fetch_list=[out])
    paddle.disable_static()
    return res


@pytest.mark.api_linalg_qr_vartype
def test_qr0():
    """
    base
    """
    A = randtool("float", -4, 4, (3, 4))
    res = np.linalg.qr(A)
    for place in places:
        for t in types:
            A = A.astype(t)
            x = paddle.to_tensor(A, dtype=t)
            dynamic_res = paddle.linalg.qr(x)
            static_res = cal_static(A, t=t, place=place)
            length = len(dynamic_res)
            for i in range(length):
                assert np.allclose(dynamic_res[i].numpy(), static_res[i])
                assert np.allclose(res[i], static_res[i])


@pytest.mark.api_linalg_qr_parameters
def test_qr1():
    """
    default
    """
    A = randtool("float", -4, 4, (4, 5))
    res = np.linalg.qr(A)
    for place in places:
        for t in types:
            A = A.astype(t)
            x = paddle.to_tensor(A, dtype=t)
            dynamic_res = paddle.linalg.qr(x)
            static_res = cal_static(A, t=t, place=place)
            length = len(dynamic_res)
            for i in range(length):
                assert np.allclose(dynamic_res[i].numpy(), static_res[i])
                assert np.allclose(res[i], static_res[i])


@pytest.mark.api_linalg_qr_parameters
def test_qr2():
    """
    x: 3d tensor
    """
    A = randtool("float", -4, 4, (3, 5, 6))
    Q, R = [], []
    for item in A:
        Q.append(np.linalg.qr(item)[0])
        R.append(np.linalg.qr(item)[1])
    for place in places:
        for t in types:
            A = A.astype(t)
            x = paddle.to_tensor(A, dtype=t)
            dynamic_res = paddle.linalg.qr(x)
            static_res = cal_static(A, t=t, place=place)
            length = len(dynamic_res[0])
            for i in range(length):
                assert np.allclose(dynamic_res[0][i].numpy(), static_res[0][i], atol=1e-2)
                assert np.allclose(dynamic_res[1][i].numpy(), static_res[1][i], atol=1e-2)
                assert np.allclose(Q[i], static_res[0][i], atol=1e-2)
                assert np.allclose(R[i], dynamic_res[1][i].numpy(), atol=1e-2)


@pytest.mark.api_linalg_qr_parameters
def test_qr3():
    """
    model = complete
    """
    A = randtool("float", -4, 4, (4, 5))
    res = np.linalg.qr(A, mode="complete")
    for place in places:
        for t in types:
            A = A.astype(t)
            x = paddle.to_tensor(A, dtype=t)
            dynamic_res = paddle.linalg.qr(x, mode="complete")
            static_res = cal_static(A, t=t, place=place, mode="complete")
            length = len(dynamic_res)
            for i in range(length):
                assert np.allclose(dynamic_res[i].numpy(), static_res[i])
                assert np.allclose(res[i], static_res[i])


@pytest.mark.api_linalg_qr_parameters
def test_qr4():
    """
    model = r
    """
    A = randtool("float", -4, 4, (4, 5))
    res = np.linalg.qr(A, mode="r")
    for place in places:
        for t in types:
            A = A.astype(t)
            x = paddle.to_tensor(A, dtype=t)
            dynamic_res = paddle.linalg.qr(x, mode="r")
            static_res = cal_static(A, t=t, place=place, mode="r")
            assert np.allclose(dynamic_res.numpy(), static_res[0])
            assert np.allclose(res, static_res[0])
