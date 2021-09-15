#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_concat.py
"""

import pytest
import numpy as np
import paddle
import paddle.fluid as fluid

# global params
types = [np.float32, np.float64, np.uint8]
if fluid.is_compiled_with_cuda() is True:
    places = [paddle.CPUPlace(), paddle.CUDAPlace(0)]
else:
    # default
    places = [paddle.CPUPlace()]


def dygraph_base(axis):
    """
    dygraph_base
    """
    for place in places:
        for t in types:
            paddle.disable_static(place)
            # paddle.set_default_dtype(t)
            x_data = np.arange(6).reshape((2, 3)).astype(t)
            y_data = np.arange(6).reshape((2, 3)).astype(t)
            x = paddle.to_tensor(x_data, stop_gradient=False)
            y = paddle.to_tensor(y_data)
            out = paddle.concat(x=(x, y), axis=axis)
            out.backward()
    return out.numpy(), x.gradient()


@pytest.mark.api_base_concat_parameters
def test_dygraph_0():
    """
    dygraph_none
    """
    out, grad = dygraph_base(0)
    res_out = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
    res_grad = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    assert np.allclose(out, res_out)
    assert np.allclose(grad, res_grad)


def static_base(axis):
    """
    static_base
    """
    for place in places:
        for t in types:
            if t != np.uint8:
                paddle.enable_static()
                main_program = fluid.Program()
                startup_program = fluid.Program()
                with fluid.program_guard(main_program=main_program, startup_program=startup_program):
                    input1 = paddle.static.data(name="x", shape=[2, 3], dtype=t)
                    input2 = paddle.static.data(name="y", shape=[2, 3], dtype=t)
                    input1.stop_gradient = False
                    output = paddle.concat(x=(input1, input2), axis=axis)
                    g = fluid.gradients(output, input1)

                    exe = fluid.Executor(place)
                    exe.run(startup_program)
                    x = np.arange(6).reshape((2, 3)).astype(t)
                    y = np.arange(6).reshape((2, 3)).astype(t)

                    out, g = exe.run(main_program, feed={"x": x, "y": y}, fetch_list=[output, g])
    return out, g


@pytest.mark.api_base_concat_parameters
def test_static_0():
    """
    static
    """
    out, grad = static_base(0)
    res_out = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
    res_grad = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    assert np.allclose(out, res_out)
    assert np.allclose(grad, res_grad)
