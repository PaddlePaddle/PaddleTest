#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_Embedding
"""
import paddle
import pytest
import numpy as np

# global params
types = [np.int32, np.int64]
if paddle.is_compiled_with_cuda() is True:
    places = [paddle.CUDAPlace(0)]
else:
    # default
    places = [paddle.CPUPlace()]


@pytest.mark.api_nn_Embedding_parameters
def test_dygraph():
    """
    test_dygraph
    """
    for place in places:
        for t in types:
            paddle.disable_static(place)
            x_data = np.array([[7, 2, 4, 5], [4, 3, 2, 9]], dtype=t)
            x = paddle.to_tensor(x_data, stop_gradient=False)
            embedding = paddle.nn.Embedding(10, 3)
            w0 = np.full(shape=(10, 3), fill_value=1).astype(np.float32)
            embedding.weight.set_value(w0)
            adam = paddle.optimizer.Adam(
                parameters=[embedding.weight], learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-08
            )
            for i in range(10):
                adam.clear_grad()
                out = embedding(x)
                out.backward()
                adam.step()
            res_weight = np.array(
                [
                    [1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0],
                    [0.90000015, 0.90000015, 0.90000015],
                    [0.90000015, 0.90000015, 0.90000015],
                    [0.90000015, 0.90000015, 0.90000015],
                    [0.90000015, 0.90000015, 0.90000015],
                    [1.0, 1.0, 1.0],
                    [0.90000015, 0.90000015, 0.90000015],
                    [1.0, 1.0, 1.0],
                    [0.90000015, 0.90000015, 0.90000015],
                ]
            )
            res_out = np.array(
                [
                    [
                        [0.91000015, 0.91000015, 0.91000015],
                        [0.91000015, 0.91000015, 0.91000015],
                        [0.91000015, 0.91000015, 0.91000015],
                        [0.91000015, 0.91000015, 0.91000015],
                    ],
                    [
                        [0.91000015, 0.91000015, 0.91000015],
                        [0.91000015, 0.91000015, 0.91000015],
                        [0.91000015, 0.91000015, 0.91000015],
                        [0.91000015, 0.91000015, 0.91000015],
                    ],
                ]
            )
            assert np.allclose(embedding.weight.numpy(), res_weight)
            assert np.allclose(out.numpy(), res_out)


@pytest.mark.api_nn_Embedding_parameters
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
                input1 = paddle.static.data(name="x", shape=[2, 4], dtype=t)
                embedding = paddle.nn.Embedding(10, 3, weight_attr=paddle.nn.initializer.Constant(value=1.0))
                adam = paddle.optimizer.Adam(
                    parameters=[embedding.weight], learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-08
                )
                output = embedding(input1)
                output = paddle.mean(output)
                adam.minimize(output)

                exe = paddle.static.Executor(place)
                exe.run(startup_program)
                x = np.array([[7, 2, 4, 5], [4, 3, 2, 9]], dtype=t)
                for i in range(10):
                    out, weight = exe.run(main_program, feed={"x": x}, fetch_list=[output, embedding.weight])
                res_weight = np.array(
                    [
                        [1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0],
                        [0.9000004, 0.9000004, 0.9000004],
                        [0.90000063, 0.90000063, 0.90000063],
                        [0.9000004, 0.9000004, 0.9000004],
                        [0.90000063, 0.90000063, 0.90000063],
                        [1.0, 1.0, 1.0],
                        [0.90000063, 0.90000063, 0.90000063],
                        [1.0, 1.0, 1.0],
                        [0.90000063, 0.90000063, 0.90000063],
                    ]
                )
                # res_out = np.array([[0.91000056]]),
                assert np.allclose(weight, res_weight)


#            assert np.allclose(out, res_out)
