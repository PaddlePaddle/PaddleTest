#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_PairwiseDistance
"""
import pytest
import numpy as np
import paddle

# global params
types = [np.float32, np.float64]
if paddle.is_compiled_with_cuda() is True:
    places = [paddle.CPUPlace(), paddle.CUDAPlace(0)]
else:
    # default
    places = [paddle.CPUPlace()]


@pytest.mark.api_nn_PairwiseDistance_vartype
def dygraph_base(p):
    """
    dygraph_base
    """
    for place in places:
        for t in types:
            paddle.disable_static(place)
            paddle.set_default_dtype(t)
            x_data = np.arange(1, 7).reshape((1, 2, 1, 3)).astype(t)
            y_data = np.arange(0, 6).reshape((1, 2, 3)).astype(t)
            x = paddle.to_tensor(x_data, stop_gradient=False)
            y = paddle.to_tensor(y_data)
            pairwise_distance = paddle.nn.PairwiseDistance(p=p)
            out = pairwise_distance(x, y)
            out.backward()
    return out.numpy(), x.gradient()


@pytest.mark.api_nn_PairwiseDistance_parameters
def test_dygraph_0_norm():
    """
    dygraph_0_norm
    """
    out, grad = dygraph_base(0)
    res_out = np.array([[[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]])
    res_grad = np.array([[[[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]]]])
    assert np.allclose(out, res_out)
    assert np.allclose(grad, res_grad)


@pytest.mark.api_nn_PairwiseDistance_parameters
def test_dygraph_1_norm():
    """
    dygraph_1_norm
    """
    out, grad = dygraph_base(1)
    res_out = np.array([[[5.0, 5.0, 5.0], [3.0, 3.0, 3.0]]])
    res_grad = np.array([[[[0.0, 0.0, 0.0]], [[1.9999981, 1.9999981, 1.9999981]]]])
    assert np.allclose(out, res_out)
    assert np.allclose(grad, res_grad)


@pytest.mark.api_nn_PairwiseDistance_parameters
def test_dygraph_2_norm():
    """
    dygraph_2_norm
    """
    out, grad = dygraph_base(2)
    res_out = np.array([[[4.1231055, 4.1231055, 4.1231055], [2.236068, 2.236068, 2.236068]]])
    res_grad = np.array([[[[-0.65189123, -0.65189123, -0.65189123]], [[1.4173558, 1.4173558, 1.4173558]]]])
    assert np.allclose(out, res_out)
    assert np.allclose(grad, res_grad)


@pytest.mark.api_nn_PairwiseDistance_parameters
def test_dygraph_positive_inf_norm():
    """
    dygraph_positive_inf_norm
    """
    out, grad = dygraph_base(np.inf)
    res_out = np.array([[[4.0, 4.0, 4.0], [2.0, 2.0, 2.0]]])
    res_grad = np.array([[[[-1.0, -1.0, -1.0]], [[1.0, 1.0, 1.0]]]])
    assert np.allclose(out, res_out)
    assert np.allclose(grad, res_grad)


@pytest.mark.api_nn_PairwiseDistance_parameters
def test_dygraph_negative_inf_norm():
    """
    dygraph_negative_inf_norm
    """
    out, grad = dygraph_base(-np.inf)
    res_out = np.array([[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]])
    res_grad = np.array([[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]])
    assert np.allclose(out, res_out)
    assert np.allclose(grad, res_grad)


@pytest.mark.api_nn_PairwiseDistance_vartype
def static_base(p):
    """
    static_base
    """
    for place in places:
        for t in types:
            paddle.enable_static()
            main_program = paddle.static.Program()
            startup_program = paddle.static.Program()
            with paddle.static.program_guard(main_program=main_program, startup_program=startup_program):
                input1 = paddle.static.data(name="x", shape=[1, 2, 1, 3], dtype=t)
                input2 = paddle.static.data(name="y", shape=[1, 2, 3], dtype=t)
                input1.stop_gradient = False
                pairwise_distance = paddle.nn.PairwiseDistance(p=p)
                output = pairwise_distance(input1, input2)
                g = paddle.static.gradients(output, input1)

                exe = paddle.static.Executor(place)
                exe.run(startup_program)
                x = np.arange(1, 7).reshape((1, 2, 1, 3)).astype(t)
                y = np.arange(0, 6).reshape((1, 2, 3)).astype(t)

                out, g = exe.run(main_program, feed={"x": x, "y": y}, fetch_list=[output, g])
    return out, g


@pytest.mark.api_nn_PairwiseDistance_parameters
def test_static_0_norm():
    """
    static_0_norm
    """
    out, grad = static_base(0)
    res_out = np.array([[[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]])
    res_grad = np.array([[[[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]]]])
    assert np.allclose(out, res_out)
    assert np.allclose(grad, res_grad)


@pytest.mark.api_nn_PairwiseDistance_parameters
def test_static_1_norm():
    """
    static_1_norm
    """
    out, grad = static_base(1)
    res_out = np.array([[[5.0, 5.0, 5.0], [3.0, 3.0, 3.0]]])
    res_grad = np.array([[[[0.0, 0.0, 0.0]], [[1.9999981, 1.9999981, 1.9999981]]]])
    assert np.allclose(out, res_out)
    assert np.allclose(grad, res_grad)


@pytest.mark.api_nn_PairwiseDistance_parameters
def test_static_2_norm():
    """
    static_2_norm
    """
    out, grad = static_base(2)
    res_out = np.array([[[4.1231055, 4.1231055, 4.1231055], [2.236068, 2.236068, 2.236068]]])
    res_grad = np.array([[[[-0.65189123, -0.65189123, -0.65189123]], [[1.4173558, 1.4173558, 1.4173558]]]])
    assert np.allclose(out, res_out)
    assert np.allclose(grad, res_grad)


@pytest.mark.api_nn_PairwiseDistance_parameters
def test_static_positive_inf_norm():
    """
    static_positive_inf_norm
    """
    out, grad = static_base(np.inf)
    res_out = np.array([[[4.0, 4.0, 4.0], [2.0, 2.0, 2.0]]])
    res_grad = np.array([[[[-1.0, -1.0, -1.0]], [[1.0, 1.0, 1.0]]]])
    assert np.allclose(out, res_out)
    assert np.allclose(grad, res_grad)


@pytest.mark.api_nn_PairwiseDistance_parameters
def test_static_negative_inf_norm():
    """
    static_negative_inf_norm
    """
    out, grad = static_base(-np.inf)
    res_out = np.array([[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]])
    res_grad = np.array([[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]])
    assert np.allclose(out, res_out)
    assert np.allclose(grad, res_grad)
