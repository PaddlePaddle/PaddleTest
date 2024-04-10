#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test to_tensor
"""
import sys
from apibase import compare

sys.path.append("../..")
import paddle
import numpy as np
import pytest
from utils.interceptor import skip_platform_is_windows

if paddle.device.is_compiled_with_cuda():
    devices = ["gpu", "cpu"]
else:
    devices = ["cpu"]


tensor_types = [
    "bool",
    "float16",
    "float32",
    "float64",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "complex64",
    "complex128",
]


@pytest.mark.api_base_to_tensor_parameters
def test_to_tensor():
    """
    x is np.array
    """
    for t in tensor_types:
        exp = np.arange(1, 7).reshape((2, 3)).astype(t)
        for d in devices:
            paddle.set_device(d)
            for s in [True, False]:
                res = paddle.to_tensor(exp, stop_gradient=s)
                compare(exp, res, delta=1e-20, rtol=1e-20)
                assert res.stop_gradient is s


@pytest.mark.api_base_to_tensor_parameters
def test_to_tensor1():
    """
    x is list
    """
    exp = [[1.1, 2.3, 4.2], [1.1, 2.3, 4.2], [1.1, 2.3, 4.2], [1.1, 2.3, 4.2]]
    for d in devices:
        paddle.set_device(d)
        for t in tensor_types:
            for s in [True, False]:
                res = paddle.to_tensor(exp, dtype=t, stop_gradient=s)
                compare(np.array(exp).astype(t), res, delta=1e-20, rtol=1e-20)
                assert res.stop_gradient is s


@pytest.mark.api_base_to_tensor_parameters
def test_to_tensor2():
    """
    x is tuple
    """
    exp = ((1.1, 2.3, 4.2), (1.1, 2.3, 4.2), (1.1, 2.3, 4.2), (1.1, 2.3, 4.2))
    for d in devices:
        paddle.set_device(d)
        for t in tensor_types:
            for s in [True, False]:
                res = paddle.to_tensor(exp, dtype=t, stop_gradient=s)
                compare(np.array(exp).astype(t), res, delta=1e-20, rtol=1e-20)
                assert res.stop_gradient is s


@pytest.mark.api_base_to_tensor_parameters
def test_to_tensor3():
    """
    x is complex list
    """
    exp = [[1.1 + 1j, 2.3, 4.2], [1.1, 2.3 - 5j, 4.2], [1.1, 2.3 + 2j, 4.2], [1.1, 2.3, 4.2]]
    for d in devices:
        paddle.set_device(d)
        for t in tensor_types:
            for s in [True, False]:
                res = paddle.to_tensor(exp, dtype=t, stop_gradient=s)
                compare(np.array(exp).astype(t), res, delta=1e-20, rtol=1e-20)
                assert res.stop_gradient is s


@skip_platform_is_windows
@pytest.mark.api_base_to_tensor_parameters
def test_to_tensor4():
    """
    test tensor.sum()
    tensor.dtype is bool
    """
    a = np.random.random((2, 2, 1, 2, 1, 1, 2, 3, 5)) * 2 - 1
    a_bool = a.astype(np.bool_)
    exp = a_bool.sum(2)
    for d in devices:
        paddle.set_device(d)
        for s in [True, False]:
            tmp = paddle.to_tensor(a_bool, stop_gradient=s)
            res = tmp.sum(2)
            compare(exp, res, delta=1e-20, rtol=1e-20)
            assert res.stop_gradient is s
            assert exp.dtype == res.numpy().dtype


# @pytest.mark.api_base_to_tensor_parameters
# def test_to_tensor5():
#     """
#     test tensor jit.to_static
#     """
#     x = paddle.to_tensor([3.2])
#
#     def jit_case5(x):
#         """test jit case"""
#         paddle.set_default_dtype("float64")
#         if paddle.device.is_compiled_with_cuda():
#             place = paddle.CUDAPlace(0)
#         else:
#             place = paddle.CPUPlace()
#         a = paddle.to_tensor([1], place=place)
#         b = paddle.to_tensor([2.1], place=place, stop_gradient=False, dtype="int64")
#         c = paddle.to_tensor([a, b, [1]], dtype="float32")
#         return c
#     exp = jit_case5(x)
#     res = paddle.jit.to_static(jit_case5)(x)
#     assert np.allclose(exp.numpy(), res.numpy())
#     assert exp.dtype == res.dtype
#     assert exp.stop_gradient == res.stop_gradient
#     assert exp.place._equals(res.place)


@pytest.mark.api_base_to_tensor_parameters
def test_to_tensor6():
    """
    test tensor jit.to_static
    """
    x = paddle.to_tensor([3.2])

    def jit_case6(x):
        """test jit case"""
        paddle.set_default_dtype("float64")
        a = paddle.to_tensor([1, 2])

        return a

    exp = jit_case6(x)
    res = paddle.jit.to_static(jit_case6)(x)
    assert np.allclose(exp.numpy(), res.numpy())
    assert exp.dtype == res.dtype
    assert exp.stop_gradient == res.stop_gradient
    assert exp.place._equals(res.place)


@pytest.mark.api_base_to_tensor_parameters
def test_to_tensor7():
    """
    test tensor jit.to_static
    """
    x = paddle.to_tensor([3.2])

    def jit_case7(x):
        """test jit case"""
        na = np.array([1, 2], dtype="int32")
        a = paddle.to_tensor(na)

        return a

    exp = jit_case7(x)
    res = paddle.jit.to_static(jit_case7)(x)
    assert np.allclose(exp.numpy(), res.numpy())
    assert exp.dtype == res.dtype
    assert exp.stop_gradient == res.stop_gradient
    assert exp.place._equals(res.place)


@pytest.mark.api_base_to_tensor_parameters
def test_to_tensor8():
    """
    test tensor jit.to_static
    """
    x = paddle.to_tensor([3.2])

    def jit_case8(x):
        """test jit case"""
        paddle.set_default_dtype("float64")
        a = paddle.to_tensor([1, 2, 3], stop_gradient=False, dtype="float32")

        return a

    exp = jit_case8(x)
    res = paddle.jit.to_static(jit_case8)(x)
    assert np.allclose(exp.numpy(), res.numpy())
    assert exp.dtype == res.dtype
    assert exp.stop_gradient == res.stop_gradient
    assert exp.place._equals(res.place)
