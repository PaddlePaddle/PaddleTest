#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test empty
"""
import paddle
import pytest
import numpy as np
import paddle.fluid as fluid
from apibase import compare

if fluid.is_compiled_with_cuda() is True:
    devices = ["gpu", "cpu"]
else:
    devices = ["cpu"]
if fluid.is_compiled_with_cuda() is True:
    places = [fluid.CPUPlace(), fluid.CUDAPlace(0)]
else:
    places = [fluid.CPUPlace()]
dtypes = [
    None,
    np.bool,
    np.float16,
    np.float32,
    np.float64,
    np.int32,
    np.int64,
    bool,
    "float16",
    "float32",
    "float64",
    "int32",
    "int64",
]
types_list = [np.int32, np.int64]


@pytest.mark.api_base_empty_vartype
@pytest.mark.api_base_empty_parameters
def test_empty_dygraph1():
    """
    x=tuple,dtype=all
    """
    paddle.disable_static()
    shape = (2, 3)
    for device in devices:
        for dtype in dtypes:
            paddle.set_device(device)
            res = paddle.empty(shape=shape, dtype=dtype)
            assert np.allclose(res.shape, shape, atol=0.005, rtol=0.05, equal_nan=True)
    paddle.enable_static()


@pytest.mark.api_base_empty_vartype
@pytest.mark.api_base_empty_parameters
def test_empty_dygraph2():
    """
    shape=list,dtype=all
    """
    paddle.disable_static()
    shape = [1, 2, 3]
    for device in devices:
        for dtype in dtypes:
            paddle.set_device(device)
            res = paddle.empty(shape=shape, dtype=dtype)
            assert np.allclose(res.shape, shape, atol=0.005, rtol=0.05, equal_nan=True)
    paddle.enable_static()


@pytest.mark.api_base_empty_vartype
@pytest.mark.api_base_empty_parameters
def test_empty_dygraph3():
    """
    shape=Tensor,dtype=all
    """
    paddle.disable_static()
    shape = paddle.to_tensor(np.array([1, 5]).astype(np.int32))
    for device in devices:
        for dtype in dtypes:
            paddle.set_device(device)
            res = paddle.empty(shape=shape, dtype=dtype)
            assert np.allclose(res.shape, [1, 5], atol=0.005, rtol=0.05, equal_nan=True)
    paddle.enable_static()


@pytest.mark.api_base_empty_vartype
@pytest.mark.api_base_empty_parameters
def test_empty_dygraph4():
    """
    # shape is a list which contains Tensor
    """
    paddle.disable_static()
    for device in devices:
        for dtype in dtypes:
            for t in types_list:
                shape1 = np.array([[[3]], [[3]]]).astype(t)
                shape2 = paddle.to_tensor(shape1)
                shape = [2, shape2]
                paddle.set_device(device)
                res = paddle.empty(shape=shape, dtype=dtype)
                np_res = np.empty(shape=[2, 3], dtype=dtype)
                assert np.allclose(res.shape, np_res.shape, atol=0.005, rtol=0.05, equal_nan=True)
    paddle.enable_static()


@pytest.mark.api_base_empty_vartype
@pytest.mark.api_base_empty_parameters
def test_empty_static5():
    """
    shape=Tensor,dtype=all
    """
    paddle.enable_static()
    for t in types_list:
        shape = np.array([3, 3, 2]).astype(t)
        feed = {"shape": shape}
        main_program = fluid.Program()
        startup_program = fluid.Program()
        with fluid.unique_name.guard():
            with fluid.program_guard(main_program=main_program, startup_program=startup_program):
                for place in places:
                    for dtype in dtypes:
                        exe = fluid.Executor(place)
                        exe.run(startup_program)
                        shape = paddle.static.data(name="shape", shape=shape.shape, dtype=t)
                        res = paddle.empty(shape=shape, dtype=dtype)
                        static_out = exe.run(fluid.default_main_program(), feed=feed, fetch_list=[res])
                        assert np.allclose(
                            np.array(static_out).shape, [1, 3, 3, 2], atol=0.005, rtol=0.05, equal_nan=True
                        )
