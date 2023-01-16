#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
/***************************************************************************
*
* Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
* @file test_Linear.py
* @author jiaxiao01
* @date 2020-09-16 14:00:00
*
**************************************************************************/
"""
import paddle
import paddle.fluid as fluid
import pytest
import numpy as np

# global params
types = [np.float32, np.float64]
if fluid.is_compiled_with_cuda() is True:
    places = [paddle.CPUPlace(), paddle.CUDAPlace(0)]
else:
    # default
    places = [paddle.CPUPlace()]


def test_dygraph():
    """
    test_dygraph
    """
    for place in places:
        for t in types:
            paddle.disable_static(place)
            paddle.set_default_dtype(t)
            x_data = np.arange(3, 9).reshape((3, 2)).astype(t)
            x = paddle.to_tensor(x_data, stop_gradient=False)
            linear = paddle.nn.Linear(2, 4)
            w0 = np.full(shape=(2, 4), fill_value=2).astype(t)
            b0 = np.full(shape=(4,), fill_value=0).astype(t)
            linear.weight.set_value(w0)
            linear.bias.set_value(b0)
            adam = paddle.optimizer.Adam(
                parameters=[linear.weight, linear.bias], learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-08
            )
            for i in range(10):
                adam.clear_grad()
                out = linear(x)
                out.backward()
                adam.step()
            res_weight = np.array(
                [[1.9000001, 1.9000001, 1.9000001, 1.9000001], [1.9000001, 1.9000001, 1.9000001, 1.9000001]]
            )
            res_bias = np.array([-0.09999976, -0.09999976, -0.09999976, -0.09999976])
            res_out = np.array(
                [
                    [13.280001, 13.280001, 13.280001, 13.280001],
                    [20.920002, 20.920002, 20.920002, 20.920002],
                    [28.560001, 28.560001, 28.560001, 28.560001],
                ]
            )
            assert np.allclose(linear.weight.numpy(), res_weight)
            assert np.allclose(linear.bias.numpy(), res_bias)
            assert np.allclose(out.numpy(), res_out)


def test_static():
    """
    test_static
    """
    for place in places:
        for t in types:
            paddle.enable_static()
            paddle.set_default_dtype(t)
            main_program = fluid.Program()
            startup_program = fluid.Program()
            with fluid.program_guard(main_program=main_program, startup_program=startup_program):
                input1 = paddle.static.data(name="x", shape=[3, 2])
                linear = paddle.nn.Linear(2, 4, weight_attr=paddle.nn.initializer.Constant(value=2.0), bias_attr=None)
                adam = paddle.optimizer.Adam(
                    parameters=[linear.weight, linear.bias], learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-08
                )
                output = linear(input1)
                output = paddle.mean(output)
                adam.minimize(output)

                exe = fluid.Executor(place)
                exe.run(startup_program)
                x = np.arange(3, 9).reshape((3, 2)).astype(t)

                for i in range(10):
                    out, weight, bias = exe.run(
                        main_program, feed={"x": x}, fetch_list=[output, linear.weight, linear.bias]
                    )
                res_weight = np.array(
                    [[1.9000002, 1.9000002, 1.9000002, 1.9000002], [1.9000002, 1.9000002, 1.9000002, 1.9000002]]
                )
                res_bias = np.array([-0.09999972, -0.09999972, -0.09999972, -0.09999972])
                res_out = (np.array([20.920002]),)
                assert np.allclose(weight, res_weight)
                assert np.allclose(bias, res_bias)
                assert np.allclose(out, res_out)
