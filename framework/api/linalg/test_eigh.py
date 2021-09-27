#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_eigh
"""

import logging
import paddle
import numpy as np
import pytest

# place = []
if paddle.device.is_compiled_with_cuda() is True:
    paddle.device.set_device("gpu:0")
    place = paddle.CUDAPlace(0)
else:
    paddle.device.set_device("cpu")
    place = paddle.CPUPlace()


def cal_eigh(dtype, x, UPLO="L"):
    """
        calculate eigh api
        """
    x = x.astype(dtype)
    input = paddle.to_tensor(x)
    dynamic_res = paddle.linalg.eigh(input, UPLO)

    paddle.enable_static()
    main_program, startup_program = paddle.static.Program(), paddle.static.Program()
    with paddle.static.program_guard(main_program=main_program, startup_program=startup_program):
        x_s = paddle.static.data(name="x_s", shape=x.shape, dtype=dtype)
        y = paddle.linalg.eigh(x_s, UPLO)
        logging.info(place)
        exe = paddle.static.Executor()
        exe.run(startup_program)
        static_res = exe.run(main_program, feed={"x_s": x}, fetch_list=[y], return_numpy=True)
        paddle.disable_static()
        length = len(dynamic_res)
        for i in range(length):
            assert np.allclose(dynamic_res[i].numpy(), static_res[i])
    # logging.info(dynamic_res)
    return static_res


@pytest.mark.api_linalg_eigh_vartype
def test_eigh_base():
    """
    base
    """
    types = ["float32", "float64", "complex64", "complex128"]
    A = np.random.rand(4, 4) * 10
    B = A + A.T
    res = np.linalg.eigh(B)
    # print(res)
    for d in types:
        e_value, e_vector = cal_eigh(d, x=B)
        # assert np.allclose(res[0], e_value)
        assert np.allclose(np.linalg.norm(res[0]), np.linalg.norm(e_value))


@pytest.mark.api_linalg_eigh_parameters
def test_eigh0():
    """
    default
    """
    types = ["float32", "float64", "complex64", "complex128"]
    A = np.random.rand(4, 4) * 10
    B = A + A.T
    res = np.linalg.eigh(B)
    # print(res)
    for d in types:
        e_value, e_vector = cal_eigh(d, x=B, UPLO="L")
        assert np.allclose(np.linalg.norm(res[0]), np.linalg.norm(e_value))


@pytest.mark.api_linalg_eigh_parameters
def test_eigh1():
    """
    UPLO='U'
    """
    types = ["float32", "float64", "complex64", "complex128"]
    A = np.random.rand(4, 4) * 10
    B = A + A.T
    res = np.linalg.eigh(B)
    # print(res)
    for d in types:
        e_value, e_vector = cal_eigh(d, x=B, UPLO="U")
        # assert np.allclose(res[0], e_value)
        assert np.allclose(np.linalg.norm(res[0]), np.linalg.norm(e_value))


@pytest.mark.api_linalg_eigh_parameters
def test_eigh2():
    """
    x: multiple dim
    """
    types = ["float32", "float64", "complex64", "complex128"]
    A = np.random.rand(4, 4, 4) * 10
    B = A + A.transpose(0, 2, 1)
    res = np.linalg.eigh(B)
    # print(res)
    for d in types:
        e_value, e_vector = cal_eigh(d, x=B)
        # assert np.allclose(res[0], e_value)
        assert np.allclose(np.linalg.norm(res[0]), np.linalg.norm(e_value))


@pytest.mark.api_linalg_eigh_parameters
def test_eigh3():
    """
    x: complex
    """
    types = ["complex64", "complex128"]
    A = np.array([[2 - 6j, 4 + 7j], [4 - 7j, 4]])
    res = np.linalg.eigh(A)
    # print(res)
    for d in types:
        e_value, e_vector = cal_eigh(d, x=A)
        # assert np.allclose(res[0], e_value)
        assert np.allclose(np.linalg.norm(res[0]), np.linalg.norm(e_value))


@pytest.mark.api_linalg_eigh_parameters
def test_eigh4():
    """
    validation eigenvector
    """
    # types = ['complex64', 'complex128']
    A = np.array([[2 - 6j, 4 + 7j], [4 - 7j, 4]])
    res = np.linalg.eigh(A)
    eigen_vector = np.array(
        [
            [(-0.749363560289441 - 0j), (0.662158783457815 + 0j)],
            [(0.32852275584841306 + 0.5749148227347227j), (0.3717884412453006 + 0.6506297721792759j)],
        ]
    )
    # print(res)
    # for d in types:
    e_value, e_vector = cal_eigh("complex128", x=A, UPLO="U")
    # assert np.allclose(res[0], e_value)
    assert np.allclose(np.linalg.norm(res[0]), np.linalg.norm(e_value))
    assert np.allclose(eigen_vector, e_vector)
