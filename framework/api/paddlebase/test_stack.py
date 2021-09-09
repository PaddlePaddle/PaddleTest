#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
# ======================================================================
#
# Copyright (c) 2017 Baidu.com, Inc. All Rights Reserved
#
# ======================================================================
"""
test_stack
"""
import pytest
import paddle
import paddle.fluid as fluid
import numpy as np
from apibase import compare


# base single tesnsor has a bug, waiting for fix.
# single tensor will report an error, it's ok.
# def test_stack1():
#     """
#     base single tesnsor has a bug, waiting for fix.
#     :return:
#     """
#     if fluid.is_compiled_with_cuda() is True:
#         places = [fluid.CPUPlace(), fluid.CUDAPlace(0)]
#     else:
#         places = [fluid.CPUPlace()]
#     types = [np.int32, np.int64, np.float32, np.float64]
#     for t in types:
#         x1 = np.array([2, 1]).astype(t)
#         for place in places:
#             paddle.disable_static(place)
#             x = paddle.to_tensor(x1)
#             res = paddle.stack(x, axis=0)
#             expect = np.stack(x1, axis=0)
#             paddle.enable_static()
#             # dygraph paddle.stack has one more dimension than np.stack, so use [expect].
#             compare(res.numpy(), [expect])


@pytest.mark.api_base_stack_vartype
def test_stack2():
    """
    base list tensor, run with dygraph.
    :return:
    """
    if fluid.is_compiled_with_cuda() is True:
        places = [fluid.CPUPlace(), fluid.CUDAPlace(0)]
    else:
        places = [fluid.CPUPlace()]
    types = [np.int32, np.int64, np.float32, np.float64]
    for t in types:
        x1 = np.array([2, 1]).astype(t)
        y1 = np.array([3, 4]).astype(t)
        for place in places:
            x = paddle.to_tensor(x1)
            y = paddle.to_tensor(y1)
            res = paddle.stack([x, y], axis=0)
            expect = np.stack([x1, y1], axis=0)
            compare(res.numpy(), expect)


@pytest.mark.api_base_stack_vartype
def test_stack3():
    """
    base tuple tensor, run with dygraph.
    :return:
    """
    if fluid.is_compiled_with_cuda() is True:
        places = [fluid.CPUPlace(), fluid.CUDAPlace(0)]
    else:
        places = [fluid.CPUPlace()]
    types = [np.int32, np.int64, np.float32, np.float64]
    for t in types:
        x1 = np.array([2, 1]).astype(t)
        y1 = np.array([3, 4]).astype(t)
        for place in places:
            x = paddle.to_tensor(x1)
            y = paddle.to_tensor(y1)
            res = paddle.stack((x, y), axis=0)
            expect = np.stack((x1, y1), axis=0)
            compare(res.numpy(), expect)


# single tensor will report an error
# def test_stack4():
#     """
#     single tensor, run with static.
#     :return:
#     """
#     if fluid.is_compiled_with_cuda() is True:
#         places = [fluid.CPUPlace(), fluid.CUDAPlace(0)]
#     else:
#         places = [fluid.CPUPlace()]
#     x = fluid.data(name="x", shape=[1, 2], dtype="float64")
#     res = paddle.stack(x, axis=0)
#     exe = fluid.Executor(fluid.CPUPlace())
#     x1 = np.array([[2, 1]]).astype("float64")
#     res_val = exe.run(fluid.default_main_program(), feed={'x': x1}, fetch_list=[res])
#     expect = np.stack(x1, axis=0)
#     result = np.allclose(res_val, expect, atol=1e-6, rtol=1e-6, equal_nan=True)
#     assert result
#     #static res_val has one more dimension than np.stack, so use [expect].
#     assert res_val[0].shape == np.array([expect]).shape


@pytest.mark.api_base_stack_vartype
def test_stack5():
    """
    list tensor, run with static.
    :return:
    """
    paddle.enable_static()
    if fluid.is_compiled_with_cuda() is True:
        places = [fluid.CPUPlace(), fluid.CUDAPlace(0)]
    else:
        places = [fluid.CPUPlace()]
    x = fluid.data(name="x", shape=[1, 2], dtype=np.float64)
    y = fluid.data(name="y", shape=[1, 2], dtype=np.float64)
    res = paddle.stack([x, y], axis=0)
    for place in places:
        exe = fluid.Executor(place)
        x1 = np.array([[2, 1]]).astype(np.float64)
        y1 = np.array([[5, 6]]).astype(np.float64)
        res_val = exe.run(fluid.default_main_program(), feed={"x": x1, "y": y1}, fetch_list=[res])
        expect = np.stack([x1, y1], axis=0)
        result = np.allclose(res_val, expect, atol=1e-6, rtol=1e-6, equal_nan=True)
        assert result
        assert res_val[0].shape == np.array(expect).shape


@pytest.mark.api_base_stack_vartype
def test_stack6():
    """
    tuple tensor, run with static.
    :return:
    """
    paddle.enable_static()
    if fluid.is_compiled_with_cuda() is True:
        places = [fluid.CPUPlace(), fluid.CUDAPlace(0)]
    else:
        places = [fluid.CPUPlace()]
    x = fluid.data(name="x", shape=[1, 2], dtype="float64")
    y = fluid.data(name="y", shape=[1, 2], dtype="float64")
    res = paddle.stack((x, y), axis=0)
    for place in places:
        exe = fluid.Executor(place)
        x1 = np.array([[2, 1]]).astype("float64")
        y1 = np.array([[5, 6]]).astype("float64")
        res_val = exe.run(fluid.default_main_program(), feed={"x": x1, "y": y1}, fetch_list=[res])
        expect = np.stack((x1, y1), axis=0)
        result = np.allclose(res_val, expect, atol=1e-6, rtol=1e-6, equal_nan=True)
        assert result
        assert res_val[0].shape == np.array(expect).shape


@pytest.mark.api_base_stack_parameters
def test_stack7():
    """
    dygraph axis between [-(R+1), R+1) dim=1, [-2,2)
    :return:
    """
    paddle.enable_static()
    if fluid.is_compiled_with_cuda() is True:
        places = [fluid.CPUPlace(), fluid.CUDAPlace(0)]
    else:
        places = [fluid.CPUPlace()]
    x1 = np.array([2, 1]).astype(np.int64)
    y1 = np.array([3, 4]).astype(np.int64)
    for place in places:
        paddle.disable_static(place)
        x = paddle.to_tensor(x1)
        y = paddle.to_tensor(y1)
        res = paddle.stack([x, y], axis=-2)
        expect = np.stack([x1, y1], axis=-2)
        paddle.enable_static()
        compare(res.numpy(), expect)
        paddle.enable_static()


@pytest.mark.api_base_stack_parameters
def test_stack8():
    """
    dygraph axis between [-(R+1), R+1) dim=1, [-2,2)
    :return:
    """
    paddle.enable_static()
    if fluid.is_compiled_with_cuda() is True:
        places = [fluid.CPUPlace(), fluid.CUDAPlace(0)]
    else:
        places = [fluid.CPUPlace()]
    x1 = np.array([2, 1]).astype(np.int64)
    y1 = np.array([3, 4]).astype(np.int64)
    for place in places:
        paddle.disable_static(place)
        x = paddle.to_tensor(x1)
        y = paddle.to_tensor(y1)
        res = paddle.stack([x, y], axis=1)
        expect = np.stack([x1, y1], axis=1)
        paddle.enable_static()
        compare(res.numpy(), expect)
        paddle.enable_static()


@pytest.mark.api_base_stack_parameters
def test_stack9():
    """
    static axis between [-(R+1), R+1) dim=1, [-2,2)
    :return:
    """
    paddle.enable_static()
    if fluid.is_compiled_with_cuda() is True:
        places = [fluid.CPUPlace(), fluid.CUDAPlace(0)]
    else:
        places = [fluid.CPUPlace()]
    x = fluid.data(name="x", shape=[1, 2], dtype="float64")
    y = fluid.data(name="y", shape=[1, 2], dtype="float64")
    res = paddle.stack((x, y), axis=-2)
    for place in places:
        exe = fluid.Executor(place)
        x1 = np.array([[2, 1]]).astype("float64")
        y1 = np.array([[5, 6]]).astype("float64")
        res_val = exe.run(fluid.default_main_program(), feed={"x": x1, "y": y1}, fetch_list=[res])
        expect = np.stack((x1, y1), axis=-2)
        result = np.allclose(res_val, expect, atol=1e-6, rtol=1e-6, equal_nan=True)
        assert result
        assert res_val[0].shape == np.array(expect).shape


@pytest.mark.api_base_stack_parameters
def test_stack10():
    """
    static axis between [-(R+1), R+1) dim=1, [-2,2)
    :return:
    """
    paddle.enable_static()
    if fluid.is_compiled_with_cuda() is True:
        places = [fluid.CPUPlace(), fluid.CUDAPlace(0)]
    else:
        places = [fluid.CPUPlace()]
    x = fluid.data(name="x", shape=[1, 2], dtype="float64")
    y = fluid.data(name="y", shape=[1, 2], dtype="float64")
    res = paddle.stack((x, y), axis=1)
    for place in places:
        exe = fluid.Executor(place)
        x1 = np.array([[2, 1]]).astype("float64")
        y1 = np.array([[5, 6]]).astype("float64")
        res_val = exe.run(fluid.default_main_program(), feed={"x": x1, "y": y1}, fetch_list=[res])
        expect = np.stack((x1, y1), axis=1)
        result = np.allclose(res_val, expect, atol=1e-6, rtol=1e-6, equal_nan=True)
        assert result
        assert res_val[0].shape == np.array(expect).shape


# 2.1 has fixed，but PR not merge into dev
# def test_stack11():
#     """
#     add nullptr check and tensor stop grad on gpu forwark backend check.
#     :return:
#     """
#     import paddle
#     import numpy as np
#     paddle.set_device("gpu")
#     x1 = paddle.to_tensor([[1.0, 2.0]])
#     x2 = paddle.to_tensor([[3.0, 4.0]])
#     x3 = paddle.to_tensor([[5.0, 6.0]])
#     x3.stop_gradient = False
#     out = paddle.stack([x1, x2, x3], axis=0)
#     assert paddle.grad(out, [x3])[-1].numpy() is not None
