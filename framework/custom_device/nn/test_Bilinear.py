#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_Bilinear
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


@pytest.mark.api_nn_Bilinear_parameters
def test_dygraph():
    """
    test_dygraph
    """
    for place in places:
        for t in types:
            paddle.disable_static(place)
            paddle.set_default_dtype(t)
            x_data = np.arange(3, 6).reshape((3, 1)).astype(t)
            y_data = np.arange(6, 12).reshape((3, 2)).astype(t)
            x = paddle.to_tensor(x_data, stop_gradient=False)
            y = paddle.to_tensor(y_data, stop_gradient=False)
            bilinear = paddle.nn.Bilinear(1, 2, 4)
            w0 = np.full(shape=(4, 1, 2), fill_value=2).astype(t)
            b0 = np.full(shape=(1, 4), fill_value=0).astype(t)
            bilinear.weight.set_value(w0)
            bilinear.bias.set_value(b0)
            adam = paddle.optimizer.Adam(
                parameters=[bilinear.weight, bilinear.bias], learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-08
            )
            for i in range(10):
                adam.clear_grad()
                out = bilinear(x, y)
                out.backward()
                adam.step()
            res_weight = np.array([[[1.9, 1.9]], [[1.9, 1.9]], [[1.9, 1.9]], [[1.9, 1.9]]])
            res_bias = np.array([[-0.09999999, -0.09999999, -0.09999999, -0.09999999]])
            res_out = np.array(
                [
                    [74.40000009, 74.40000009, 74.40000009, 74.40000009],
                    [129.79000015, 129.79000015, 129.79000015, 129.79000015],
                    [200.46000023, 200.46000023, 200.46000023, 200.46000023],
                ]
            )
            assert np.allclose(bilinear.weight.numpy(), res_weight)
            assert np.allclose(bilinear.bias.numpy(), res_bias)
            assert np.allclose(out.numpy(), res_out)


@pytest.mark.api_nn_Bilinear_parameters
def test_static():
    """
    test_static
    """
    for place in places:
        for t in types:
            paddle.enable_static()
            paddle.set_default_dtype(t)
            main_program = paddle.static.Program()
            startup_program = paddle.static.Program()
            with paddle.static.program_guard(main_program=main_program, startup_program=startup_program):
                input1 = paddle.static.data(name="x", shape=[3, 1])
                input2 = paddle.static.data(name="y", shape=[3, 2])
                bilinear = paddle.nn.Bilinear(
                    1, 2, 4, weight_attr=paddle.nn.initializer.Constant(value=2.0), bias_attr=None
                )
                adam = paddle.optimizer.Adam(
                    parameters=[bilinear.weight, bilinear.bias],
                    learning_rate=0.01,
                    beta1=0.9,
                    beta2=0.999,
                    epsilon=1e-08,
                )
                output = bilinear(input1, input2)
                output = paddle.mean(output)
                adam.minimize(output)

                exe = paddle.static.Executor(place)
                exe.run(startup_program)
                x = np.arange(3, 6).reshape((3, 1)).astype(t)
                y = np.arange(6, 12).reshape((3, 2)).astype(t)

                for i in range(10):
                    out, weight, bias = exe.run(
                        main_program, feed={"x": x, "y": y}, fetch_list=[output, bilinear.weight, bilinear.bias]
                    )
                res_weight = np.array(
                    [
                        [[1.9000002, 1.9000002]],
                        [[1.9000002, 1.9000002]],
                        [[1.9000002, 1.9000002]],
                        [[1.9000002, 1.9000002]],
                    ]
                )
                res_bias = np.array([[-0.09999972, -0.09999972, -0.09999972, -0.09999972]])
                res_out = (np.array([134.88336]),)
                assert np.allclose(weight, res_weight)
                assert np.allclose(bias, res_bias)
                assert np.allclose(out, res_out)
