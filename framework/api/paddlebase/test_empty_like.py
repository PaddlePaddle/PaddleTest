#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test empty_like
"""
import paddle
import pytest
import numpy as np


if paddle.is_compiled_with_cuda() is True:
    devices = ["gpu", "cpu"]
else:
    devices = ["cpu"]
if paddle.is_compiled_with_cuda() is True:
    places = [paddle.CPUPlace(), paddle.CUDAPlace(0)]
else:
    places = [paddle.CPUPlace()]
types_list = [np.bool_, np.float16, np.float32, np.float64, np.int32, np.int64]
types_list_cpu = [np.bool_, np.float32, np.float64, np.int32, np.int64]
dtypes = [
    None,
    # np.bool_,
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


@pytest.mark.api_base_empty_vartype
@pytest.mark.api_base_empty_parameters
def test_empty_like_dygraph1():
    """
    x=Tensor,dtype=all
    """
    paddle.disable_static()
    for device in devices:
        for dtype in dtypes:
            for t in types_list:
                paddle.set_device(device)
                x = paddle.to_tensor(np.array([3, 2]).astype(t))
                res = paddle.empty_like(x=x, dtype=dtype)
                assert np.allclose(res.shape, x.shape, atol=0.005, rtol=0.05, equal_nan=True)
    paddle.enable_static()


@pytest.mark.api_base_empty_vartype
@pytest.mark.api_base_empty_parameters
def test_empty_like_static1():
    """
    x=Tensor,dtype=all
    """
    for t in types_list_cpu:
        for place in places:
            for dtype in dtypes:
                main_program = paddle.static.Program()
                startup_program = paddle.static.Program()
                x = np.array([3, 3, 2]).astype(t)
                feed = {"x": x}
                with paddle.utils.unique_name.guard():
                    with paddle.static.program_guard(main_program=main_program, startup_program=startup_program):
                        x = paddle.static.data(name="x", shape=x.shape, dtype=t)
                        exe = paddle.static.Executor(place)
                        exe.run(startup_program)
                        res = paddle.empty_like(x=x, dtype=dtype)
                        static_out = exe.run(paddle.static.default_main_program(), feed=feed, fetch_list=[res])
                        assert np.allclose(np.array(static_out).shape, [1, 3], atol=0.005, rtol=0.05, equal_nan=True)
