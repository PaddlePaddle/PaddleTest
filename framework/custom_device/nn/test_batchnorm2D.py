#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_batchnorm2D
"""
import pytest
import numpy as np
import paddle.static as static
import paddle
from apibase import compare

seed = 33
types = [np.float32, np.float64]
if paddle.device.is_compiled_with_cuda() is True:
    places = [paddle.CPUPlace(), paddle.CUDAPlace(0)]
else:
    places = [paddle.CPUPlace()]


@pytest.mark.api_nn_BatchNorm2D_parameters
def test_dygraph1():
    """
    input=4-D Tensor
    :return:
    """
    for place in places:
        for t in types:
            if t == np.float64:
                paddle.set_default_dtype("float64")
            elif t == np.float32:
                paddle.set_default_dtype("float32")

            paddle.disable_static(place)
            x_data = np.array(
                [
                    [[[0.6964692, 0.28613934, 0.22685145], [0.5513148, 0.71946895, 0.42310646]]],
                    [[[0.9807642, 0.6848297, 0.4809319], [0.39211753, 0.343178, 0.7290497]]],
                ]
            ).astype(t)
            x = paddle.to_tensor(x_data)
            batch_norm = paddle.nn.BatchNorm2D(1)
            batch_norm_out = batch_norm(x)
            res = np.array(
                [
                    [[[0.71878886, -1.2011799, -1.478593], [0.03959884, 0.8264067, -0.5602989]]],
                    [[[2.0490298, 0.6643267, -0.28972864], [-0.7052988, -0.93429106, 0.8712358]]],
                ]
            )
            compare(res, batch_norm_out.numpy())
            paddle.enable_static()


@pytest.mark.api_nn_BatchNorm2D_parameters
def test_dygraph2():
    """
    input=4-D Tensor, momentum=0.1.
    :return:
    """
    for place in places:
        for t in types:
            if t == np.float64:
                paddle.set_default_dtype("float64")
            elif t == np.float32:
                paddle.set_default_dtype("float32")

            paddle.disable_static(place)
            x_data = np.array(
                [
                    [[[0.6964692, 0.28613934, 0.22685145], [0.5513148, 0.71946895, 0.42310646]]],
                    [[[0.9807642, 0.6848297, 0.4809319], [0.39211753, 0.343178, 0.7290497]]],
                ]
            ).astype(t)
            x = paddle.to_tensor(x_data)
            batch_norm = paddle.nn.BatchNorm2D(num_features=1, momentum=0.1)
            batch_norm_out = batch_norm(x)
            res = np.array(
                [
                    [[[0.71878886, -1.2011799, -1.478593], [0.03959884, 0.8264067, -0.5602989]]],
                    [[[2.0490298, 0.6643267, -0.28972864], [-0.7052988, -0.93429106, 0.8712358]]],
                ]
            )
            compare(res, batch_norm_out.numpy())
            paddle.enable_static()


@pytest.mark.api_nn_BatchNorm2D_parameters
def test_dygraph3():
    """
    input=4-D Tensor, momentum=0.1, epsilon<=1e-3
    :return:
    """
    for place in places:
        for t in types:
            if t == np.float64:
                paddle.set_default_dtype("float64")
            elif t == np.float32:
                paddle.set_default_dtype("float32")

            paddle.disable_static(place)
            x_data = np.array(
                [
                    [[[0.6964692, 0.28613934, 0.22685145], [0.5513148, 0.71946895, 0.42310646]]],
                    [[[0.9807642, 0.6848297, 0.4809319], [0.39211753, 0.343178, 0.7290497]]],
                ]
            ).astype(t)
            x = paddle.to_tensor(x_data)
            batch_norm = paddle.nn.BatchNorm2D(num_features=1, momentum=0.1, epsilon=1e-5)
            batch_norm_out = batch_norm(x)
            res = np.array(
                [
                    [[[0.71878886, -1.2011799, -1.478593], [0.03959884, 0.8264067, -0.5602989]]],
                    [[[2.0490298, 0.6643267, -0.28972864], [-0.7052988, -0.93429106, 0.8712358]]],
                ]
            )
            compare(res, batch_norm_out.numpy())
            paddle.enable_static()


@pytest.mark.api_nn_BatchNorm2D_parameters
def test_dygraph4():
    """
    input=4-D Tensor, momentum=0.1, epsilon<=1e-3, data_format='NCHW'
    :return:
    """
    for place in places:
        for t in types:
            if t == np.float64:
                paddle.set_default_dtype("float64")
            elif t == np.float32:
                paddle.set_default_dtype("float32")

            paddle.disable_static(place)
            x_data = np.array(
                [
                    [[[0.6964692, 0.28613934, 0.22685145], [0.5513148, 0.71946895, 0.42310646]]],
                    [[[0.9807642, 0.6848297, 0.4809319], [0.39211753, 0.343178, 0.7290497]]],
                ]
            ).astype(t)
            x = paddle.to_tensor(x_data)
            batch_norm = paddle.nn.BatchNorm2D(num_features=1, momentum=0.1, epsilon=1e-5, data_format="NCHW")
            batch_norm_out = batch_norm(x)
            res = np.array(
                [
                    [[[0.71878886, -1.2011799, -1.478593], [0.03959884, 0.8264067, -0.5602989]]],
                    [[[2.0490298, 0.6643267, -0.28972864], [-0.7052988, -0.93429106, 0.8712358]]],
                ]
            )
            compare(res, batch_norm_out.numpy())
            paddle.enable_static()


@pytest.mark.api_nn_BatchNorm2D_parameters
def test_dygraph5():
    """
    input=4-D Tensor, momentum=0.1, epsilon<=1e-3, data_format='NCHW'
    :return:
    """
    for place in places:
        for t in types:
            if t == np.float64:
                paddle.set_default_dtype("float64")
            elif t == np.float32:
                paddle.set_default_dtype("float32")

            paddle.disable_static(place)
            x_data = np.array(
                [
                    [[[0.6964692, 0.28613934, 0.22685145], [0.5513148, 0.71946895, 0.42310646]]],
                    [[[0.9807642, 0.6848297, 0.4809319], [0.39211753, 0.343178, 0.7290497]]],
                ]
            ).astype(t)
            x = paddle.to_tensor(x_data)
            batch_norm = paddle.nn.BatchNorm2D(num_features=1, momentum=0.1, epsilon=1e-5, data_format="NCHW")
            batch_norm_out = batch_norm(x)
            res = np.array(
                [
                    [[[0.71878886, -1.2011799, -1.478593], [0.03959884, 0.8264067, -0.5602989]]],
                    [[[2.0490298, 0.6643267, -0.28972864], [-0.7052988, -0.93429106, 0.8712358]]],
                ]
            )
            compare(res, batch_norm_out.numpy())
            paddle.enable_static()


@pytest.mark.api_nn_BatchNorm2D_parameters
def test_dygraph6():
    """
    input=4-D Tensor, momentum=0.1, epsilon<=1e-3, data_format='NCHW', weight_attr=False
    :return:
    """
    for place in places:
        for t in types:
            if t == np.float64:
                paddle.set_default_dtype("float64")
            elif t == np.float32:
                paddle.set_default_dtype("float32")

            paddle.disable_static(place)
            w_param_attrs = paddle.ParamAttr(
                name="w1_weight", learning_rate=0.5, regularizer=paddle.regularizer.L2Decay(1.0), trainable=True
            )
            b_param_attrs = paddle.ParamAttr(
                name="b1_weight", learning_rate=0.5, regularizer=paddle.regularizer.L2Decay(1.0), trainable=True
            )
            x_data = np.array(
                [
                    [[[0.6964692, 0.28613934, 0.22685145], [0.5513148, 0.71946895, 0.42310646]]],
                    [[[0.9807642, 0.6848297, 0.4809319], [0.39211753, 0.343178, 0.7290497]]],
                ]
            ).astype(t)
            x = paddle.to_tensor(x_data)
            batch_norm = paddle.nn.BatchNorm2D(
                num_features=1,
                momentum=0.1,
                epsilon=1e-5,
                data_format="NCHW",
                weight_attr=w_param_attrs,
                bias_attr=b_param_attrs,
            )
            batch_norm_out = batch_norm(x)
            res = np.array(
                [
                    [[[0.71878886, -1.2011799, -1.478593], [0.03959884, 0.8264067, -0.5602989]]],
                    [[[2.0490298, 0.6643267, -0.28972864], [-0.7052988, -0.93429106, 0.8712358]]],
                ]
            )
            compare(res, batch_norm_out.numpy())
            paddle.enable_static()


@pytest.mark.api_nn_BatchNorm2D_parameters
def test_dygraph7():
    """
    input=4-D Tensor, momentum=0.1, epsilon<=1e-3, data_format='NCHW', weight_attr=False, bias_attr=b_param_attrs
    :return:
    """
    for place in places:
        for t in types:
            if t == np.float64:
                paddle.set_default_dtype("float64")
            elif t == np.float32:
                paddle.set_default_dtype("float32")

            paddle.disable_static(place)
            b_param_attrs = paddle.ParamAttr(
                name="b1_weight", learning_rate=0.5, regularizer=paddle.regularizer.L2Decay(1.0), trainable=True
            )
            x_data = np.array(
                [
                    [[[0.6964692, 0.28613934, 0.22685145], [0.5513148, 0.71946895, 0.42310646]]],
                    [[[0.9807642, 0.6848297, 0.4809319], [0.39211753, 0.343178, 0.7290497]]],
                ]
            ).astype(t)
            x = paddle.to_tensor(x_data)
            batch_norm = paddle.nn.BatchNorm2D(
                num_features=1,
                momentum=0.1,
                epsilon=1e-5,
                data_format="NCHW",
                weight_attr=False,
                bias_attr=b_param_attrs,
            )
            batch_norm_out = batch_norm(x)
            res = np.array(
                [
                    [[[0.71878886, -1.2011799, -1.478593], [0.03959884, 0.8264067, -0.5602989]]],
                    [[[2.0490298, 0.6643267, -0.28972864], [-0.7052988, -0.93429106, 0.8712358]]],
                ]
            )
            compare(res, batch_norm_out.numpy())
            paddle.enable_static()


@pytest.mark.api_nn_BatchNorm2D_parameters
def test_static1():
    """
    input=4-D Tensor
    :return:
    """
    for place in places:
        for t in types:
            if t == np.float64:
                paddle.set_default_dtype("float64")
            elif t == np.float32:
                paddle.set_default_dtype("float32")

            main_program = static.Program()
            startup_program = static.Program()
            main_program.random_seed = seed
            startup_program.random_seed = seed
            x_data = np.array(
                [
                    [[[0.6964692, 0.28613934, 0.22685145], [0.5513148, 0.71946895, 0.42310646]]],
                    [[[0.9807642, 0.6848297, 0.4809319], [0.39211753, 0.343178, 0.7290497]]],
                ]
            ).astype(t)
            feed = {"x_data": x_data}
            with paddle.utils.unique_name.guard():
                with static.program_guard(main_program=main_program, startup_program=startup_program):
                    x_data = static.data(name="x_data", shape=[2, 1, 2, 3], dtype=t)
                    exe = static.Executor(place)
                    output = paddle.nn.BatchNorm2D(1)(x_data)
                    exe.run(startup_program)
                    res = exe.run(main_program, feed=feed, fetch_list=[output])
                    expect = np.array(
                        [
                            [[[0.71878886, -1.2011799, -1.478593], [0.03959884, 0.8264067, -0.5602989]]],
                            [[[2.0490298, 0.6643267, -0.28972864], [-0.7052988, -0.93429106, 0.8712358]]],
                        ]
                    )
                    compare(res[0], expect)


@pytest.mark.api_nn_BatchNorm2D_parameters
def test_static2():
    """
    input=4-D Tensor, momentum=0.1
    :return:
    """
    for place in places:
        for t in types:
            if t == np.float64:
                paddle.set_default_dtype("float64")
            elif t == np.float32:
                paddle.set_default_dtype("float32")

            main_program = static.Program()
            startup_program = static.Program()
            main_program.random_seed = seed
            startup_program.random_seed = seed
            x_data = np.array(
                [
                    [[[0.6964692, 0.28613934, 0.22685145], [0.5513148, 0.71946895, 0.42310646]]],
                    [[[0.9807642, 0.6848297, 0.4809319], [0.39211753, 0.343178, 0.7290497]]],
                ]
            ).astype(t)
            feed = {"x_data": x_data}
            with paddle.utils.unique_name.guard():
                with static.program_guard(main_program=main_program, startup_program=startup_program):
                    x_data = static.data(name="x_data", shape=[2, 1, 2, 3], dtype=t)
                    exe = static.Executor(place)
                    output = paddle.nn.BatchNorm2D(num_features=1, momentum=0.1)(x_data)
                    exe.run(startup_program)
                    res = exe.run(main_program, feed=feed, fetch_list=[output])
                    expect = np.array(
                        [
                            [[[0.71878886, -1.2011799, -1.478593], [0.03959884, 0.8264067, -0.5602989]]],
                            [[[2.0490298, 0.6643267, -0.28972864], [-0.7052988, -0.93429106, 0.8712358]]],
                        ]
                    )
                    compare(res[0], expect)


@pytest.mark.api_nn_BatchNorm2D_parameters
def test_static3():
    """
    input=4-D Tensor, momentum=0.1, epsilon<=1e-3
    :return:
    """
    for place in places:
        for t in types:
            if t == np.float64:
                paddle.set_default_dtype("float64")
            elif t == np.float32:
                paddle.set_default_dtype("float32")

            main_program = static.Program()
            startup_program = static.Program()
            main_program.random_seed = seed
            startup_program.random_seed = seed
            x_data = np.array(
                [
                    [[[0.6964692, 0.28613934, 0.22685145], [0.5513148, 0.71946895, 0.42310646]]],
                    [[[0.9807642, 0.6848297, 0.4809319], [0.39211753, 0.343178, 0.7290497]]],
                ]
            ).astype(t)
            feed = {"x_data": x_data}
            with paddle.utils.unique_name.guard():
                with static.program_guard(main_program=main_program, startup_program=startup_program):
                    x_data = static.data(name="x_data", shape=[2, 1, 2, 3], dtype=t)
                    exe = static.Executor(place)
                    output = paddle.nn.BatchNorm2D(num_features=1, momentum=0.1, epsilon=1e-5)(x_data)
                    exe.run(startup_program)
                    res = exe.run(main_program, feed=feed, fetch_list=[output])
                    expect = np.array(
                        [
                            [[[0.71878886, -1.2011799, -1.478593], [0.03959884, 0.8264067, -0.5602989]]],
                            [[[2.0490298, 0.6643267, -0.28972864], [-0.7052988, -0.93429106, 0.8712358]]],
                        ]
                    )
                    compare(res[0], expect)


@pytest.mark.api_nn_BatchNorm2D_parameters
def test_static4():
    """
    input=4-D Tensor, momentum=0.1, epsilon<=1e-3, data_format='NCHW'
    :return:
    """
    for place in places:
        for t in types:
            if t == np.float64:
                paddle.set_default_dtype("float64")
            elif t == np.float32:
                paddle.set_default_dtype("float32")

            main_program = static.Program()
            startup_program = static.Program()
            main_program.random_seed = seed
            startup_program.random_seed = seed
            x_data = np.array(
                [
                    [[[0.6964692, 0.28613934, 0.22685145], [0.5513148, 0.71946895, 0.42310646]]],
                    [[[0.9807642, 0.6848297, 0.4809319], [0.39211753, 0.343178, 0.7290497]]],
                ]
            ).astype(t)
            feed = {"x_data": x_data}
            with paddle.utils.unique_name.guard():
                with static.program_guard(main_program=main_program, startup_program=startup_program):
                    x_data = static.data(name="x_data", shape=[2, 1, 2, 3], dtype=t)
                    exe = static.Executor(place)
                    output = paddle.nn.BatchNorm2D(num_features=1, momentum=0.1, epsilon=1e-5, data_format="NCHW")(
                        x_data
                    )
                    exe.run(startup_program)
                    res = exe.run(main_program, feed=feed, fetch_list=[output])
                    expect = np.array(
                        [
                            [[[0.71878886, -1.2011799, -1.478593], [0.03959884, 0.8264067, -0.5602989]]],
                            [[[2.0490298, 0.6643267, -0.28972864], [-0.7052988, -0.93429106, 0.8712358]]],
                        ]
                    )
                    compare(res[0], expect)


@pytest.mark.api_nn_BatchNorm2D_parameters
def test_static5():
    """
    input=4-D Tensor, momentum=0.1, epsilon<=1e-3, data_format='NCWH'
    :return:
    """
    for place in places:
        for t in types:
            if t == np.float64:
                paddle.set_default_dtype("float64")
            elif t == np.float32:
                paddle.set_default_dtype("float32")

            main_program = static.Program()
            startup_program = static.Program()
            main_program.random_seed = seed
            startup_program.random_seed = seed
            x_data = np.array(
                [
                    [[[0.6964692, 0.28613934, 0.22685145], [0.5513148, 0.71946895, 0.42310646]]],
                    [[[0.9807642, 0.6848297, 0.4809319], [0.39211753, 0.343178, 0.7290497]]],
                ]
            ).astype(t)
            feed = {"x_data": x_data}
            with paddle.utils.unique_name.guard():
                with static.program_guard(main_program=main_program, startup_program=startup_program):
                    x_data = static.data(name="x_data", shape=[2, 1, 2, 3], dtype=t)
                    exe = static.Executor(place)
                    output = paddle.nn.BatchNorm2D(num_features=1, momentum=0.1, epsilon=1e-5, data_format="NCHW")(
                        x_data
                    )
                    exe.run(startup_program)
                    res = exe.run(main_program, feed=feed, fetch_list=[output])
                    expect = np.array(
                        [
                            [[[0.71878886, -1.2011799, -1.478593], [0.03959884, 0.8264067, -0.5602989]]],
                            [[[2.0490298, 0.6643267, -0.28972864], [-0.7052988, -0.93429106, 0.8712358]]],
                        ]
                    )
                    compare(res[0], expect)


@pytest.mark.api_nn_BatchNorm2D_parameters
def test_static6():
    """
    input=4-D Tensor, momentum=0.1, epsilon<=1e-3, data_format='NCHW', weight_attr=w, bias_attr=b
    :return:
    """
    for place in places:
        for t in types:
            if t == np.float64:
                paddle.set_default_dtype("float64")
            elif t == np.float32:
                paddle.set_default_dtype("float32")

            main_program = static.Program()
            startup_program = static.Program()
            main_program.random_seed = seed
            startup_program.random_seed = seed
            x_data = np.array(
                [
                    [[[0.6964692, 0.28613934, 0.22685145], [0.5513148, 0.71946895, 0.42310646]]],
                    [[[0.9807642, 0.6848297, 0.4809319], [0.39211753, 0.343178, 0.7290497]]],
                ]
            ).astype(t)
            feed = {"x_data": x_data}
            with paddle.utils.unique_name.guard():
                with static.program_guard(main_program=main_program, startup_program=startup_program):
                    x_data = static.data(name="x_data", shape=[2, 1, 2, 3], dtype=t)
                    w_param_attrs = paddle.ParamAttr(
                        name="w1_weight", learning_rate=0.5, regularizer=paddle.regularizer.L2Decay(1.0), trainable=True
                    )
                    b_param_attrs = paddle.ParamAttr(
                        name="b1_weight", learning_rate=0.5, regularizer=paddle.regularizer.L2Decay(1.0), trainable=True
                    )
                    exe = static.Executor(place)
                    output = paddle.nn.BatchNorm2D(
                        num_features=1,
                        momentum=0.1,
                        epsilon=1e-5,
                        data_format="NCHW",
                        weight_attr=w_param_attrs,
                        bias_attr=b_param_attrs,
                    )(x_data)
                    exe.run(startup_program)
                    res = exe.run(main_program, feed=feed, fetch_list=[output])
                    expect = np.array(
                        [
                            [[[0.71878886, -1.2011799, -1.478593], [0.03959884, 0.8264067, -0.5602989]]],
                            [[[2.0490298, 0.6643267, -0.28972864], [-0.7052988, -0.93429106, 0.8712358]]],
                        ]
                    )
                    compare(res[0], expect)


@pytest.mark.api_nn_BatchNorm2D_parameters
def test_static7():
    """
    input=4-D Tensor, momentum=0.1, epsilon<=1e-3, data_format='NCHW', weight_attr=False, bias_attr=b
    :return:
    """
    for place in places:
        for t in types:
            if t == np.float64:
                paddle.set_default_dtype("float64")
            elif t == np.float32:
                paddle.set_default_dtype("float32")

            main_program = static.Program()
            startup_program = static.Program()
            main_program.random_seed = seed
            startup_program.random_seed = seed
            x_data = np.array(
                [
                    [[[0.6964692, 0.28613934, 0.22685145], [0.5513148, 0.71946895, 0.42310646]]],
                    [[[0.9807642, 0.6848297, 0.4809319], [0.39211753, 0.343178, 0.7290497]]],
                ]
            ).astype(t)
            feed = {"x_data": x_data}
            with paddle.utils.unique_name.guard():
                with static.program_guard(main_program=main_program, startup_program=startup_program):
                    x_data = static.data(name="x_data", shape=[2, 1, 2, 3], dtype=t)
                    b_param_attrs = paddle.ParamAttr(
                        name="b1_weight", learning_rate=0.5, regularizer=paddle.regularizer.L2Decay(1.0), trainable=True
                    )
                    exe = static.Executor(place)
                    output = paddle.nn.BatchNorm2D(
                        num_features=1,
                        momentum=0.1,
                        epsilon=1e-5,
                        data_format="NCHW",
                        weight_attr=False,
                        bias_attr=b_param_attrs,
                    )(x_data)
                    exe.run(startup_program)
                    res = exe.run(main_program, feed=feed, fetch_list=[output])
                    expect = np.array(
                        [
                            [[[0.71878886, -1.2011799, -1.478593], [0.03959884, 0.8264067, -0.5602989]]],
                            [[[2.0490298, 0.6643267, -0.28972864], [-0.7052988, -0.93429106, 0.8712358]]],
                        ]
                    )
                    compare(res[0], expect)
