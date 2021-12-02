#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_Dropout2D
"""
import paddle
import pytest
import numpy as np

# global params
types = [np.float32, np.float64]
if paddle.is_compiled_with_cuda() is True:
    places = [paddle.CPUPlace(), paddle.CUDAPlace(0)]
else:
    # default
    places = [paddle.CPUPlace()]


@pytest.mark.api_nn_Dropout2_parameters
def test_dygraph():
    """
    test_dygraph
    """
    for place in places:
        for t in types:
            paddle.disable_static(place)
            x_data = np.arange(1, 13).reshape(2, 1, 2, 3).astype(t)
            x = paddle.to_tensor(x_data, stop_gradient=False)
            dropout2D = paddle.nn.Dropout2D(p=0.5)
            for i in range(1):
                out = dropout2D(x)
                out.backward()
            index = np.flatnonzero(out)
            for i in index:
                assert np.allclose(x_data.flatten()[i] * 2, out.numpy().flatten()[i])
                assert np.allclose(x.gradient().flatten()[i], 2)


@pytest.mark.api_nn_Dropout2_parameters
def test_static():
    """
    test_static
    """
    for place in places:
        for t in types:
            paddle.enable_static()
            main_program = paddle.static.Program()
            startup_program = paddle.static.Program()
            with paddle.static.program_guard(main_program=main_program, startup_program=startup_program):
                input1 = paddle.static.data(name="x", shape=[2, 1, 2, 3], dtype=t)
                input1.stop_gradient = False
                dropout2D = paddle.nn.Dropout2D(p=0.5)
                output = dropout2D(input1)
                g = paddle.static.gradients(output, input1)

                exe = paddle.static.Executor(place)
                exe.run(startup_program)
                x = np.arange(1, 13).reshape(2, 1, 2, 3).astype(t)

                for i in range(1):
                    out, g = exe.run(main_program, feed={"x": x}, fetch_list=[output, g])

                index = np.flatnonzero(out)
                for i in index:
                    assert np.allclose(x.flatten()[i] * 2, out.flatten()[i])
                    assert np.allclose(g.flatten()[i], 2)
