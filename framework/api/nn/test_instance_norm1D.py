#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_instancenorm1d
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


@pytest.mark.api_nn_InstanceNorm1D_parameters
def test_dygraph1():
    """
    input=3-D Tensor
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
                    [[0.6964692, 0.28613934, 0.22685145], [0.5513148, 0.71946895, 0.42310646]],
                    [[0.9807642, 0.6848297, 0.4809319], [0.39211753, 0.343178, 0.7290497]],
                ]
            ).astype(t)
            x = paddle.to_tensor(x_data)
            batch_norm = paddle.nn.InstanceNorm1D(2)
            batch_norm_out = batch_norm(x)
            res = np.array(
                [
                    [[1.4045197, -0.5603124, -0.8442078], [-0.10968472, 1.2754805, -1.165797]],
                    [[1.2924801, -0.1494856, -1.1429948], [-0.55954623, -0.84480286, 1.4043493]],
                ]
            )
            compare(res, batch_norm_out.numpy())
            paddle.enable_static()


@pytest.mark.api_nn_InstanceNorm1D_parameters
def test_dygraph2():
    """
    input=3-D Tensor, epsilon<=1e-03
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
                    [[0.6964692, 0.28613934, 0.22685145], [0.5513148, 0.71946895, 0.42310646]],
                    [[0.9807642, 0.6848297, 0.4809319], [0.39211753, 0.343178, 0.7290497]],
                ]
            ).astype(t)
            x = paddle.to_tensor(x_data)
            batch_norm = paddle.nn.InstanceNorm1D(num_features=2, epsilon=1e-05)
            batch_norm_out = batch_norm(x)
            res = np.array(
                [
                    [[1.4045197, -0.5603124, -0.8442078], [-0.10968472, 1.2754805, -1.165797]],
                    [[1.2924801, -0.1494856, -1.1429948], [-0.55954623, -0.84480286, 1.4043493]],
                ]
            )
            compare(res, batch_norm_out.numpy())
            paddle.enable_static()


@pytest.mark.api_nn_InstanceNorm1D_parameters
def test_dygraph3():
    """
    input=3-D Tensor, epsilon<=1e-03, momentum=0.1
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
                    [[0.6964692, 0.28613934, 0.22685145], [0.5513148, 0.71946895, 0.42310646]],
                    [[0.9807642, 0.6848297, 0.4809319], [0.39211753, 0.343178, 0.7290497]],
                ]
            ).astype(t)
            x = paddle.to_tensor(x_data)
            batch_norm = paddle.nn.InstanceNorm1D(num_features=2, epsilon=1e-05, momentum=0.1)
            batch_norm_out = batch_norm(x)
            res = np.array(
                [
                    [[1.4045197, -0.5603124, -0.8442078], [-0.10968472, 1.2754805, -1.165797]],
                    [[1.2924801, -0.1494856, -1.1429948], [-0.55954623, -0.84480286, 1.4043493]],
                ]
            )
            compare(res, batch_norm_out.numpy())
            paddle.enable_static()


@pytest.mark.api_nn_InstanceNorm1D_parameters
def test_dygraph4():
    """
    input=3-D Tensor, epsilon<=1e-03, momentum=0.1, weight_attr=w_param_attrs, bias_attr=b_param_attrs
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
                    [[0.6964692, 0.28613934, 0.22685145], [0.5513148, 0.71946895, 0.42310646]],
                    [[0.9807642, 0.6848297, 0.4809319], [0.39211753, 0.343178, 0.7290497]],
                ]
            ).astype(t)
            w_param_attrs = paddle.ParamAttr(
                name="w_weight", learning_rate=0.5, regularizer=paddle.regularizer.L2Decay(1.0), trainable=True
            )
            b_param_attrs = paddle.ParamAttr(
                name="b_weight", learning_rate=0.5, regularizer=paddle.regularizer.L2Decay(1.0), trainable=True
            )
            x = paddle.to_tensor(x_data)
            batch_norm = paddle.nn.InstanceNorm1D(
                num_features=2, epsilon=1e-05, momentum=0.1, weight_attr=w_param_attrs, bias_attr=b_param_attrs
            )
            batch_norm_out = batch_norm(x)
            res = np.array(
                [
                    [[1.4045197, -0.5603124, -0.8442078], [-0.10968472, 1.2754805, -1.165797]],
                    [[1.2924801, -0.1494856, -1.1429948], [-0.55954623, -0.84480286, 1.4043493]],
                ]
            )
            compare(res, batch_norm_out.numpy())
            paddle.enable_static()


@pytest.mark.api_nn_InstanceNorm1D_parameters
def test_dygraph5():
    """
    input=3-D Tensor, epsilon<=1e-03, momentum=0.1, data_format='NCL'
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
                    [[0.6964692, 0.28613934, 0.22685145], [0.5513148, 0.71946895, 0.42310646]],
                    [[0.9807642, 0.6848297, 0.4809319], [0.39211753, 0.343178, 0.7290497]],
                ]
            ).astype(t)
            x = paddle.to_tensor(x_data)
            batch_norm = paddle.nn.InstanceNorm1D(num_features=2, epsilon=1e-05, momentum=0.1, data_format="NCL")
            batch_norm_out = batch_norm(x)
            res = np.array(
                [
                    [[1.4045197, -0.5603124, -0.8442078], [-0.10968472, 1.2754805, -1.165797]],
                    [[1.2924801, -0.1494856, -1.1429948], [-0.55954623, -0.84480286, 1.4043493]],
                ]
            )
            compare(res, batch_norm_out.numpy())
            paddle.enable_static()


@pytest.mark.api_nn_InstanceNorm1D_parameters
def test_dygraph6():
    """
    input=3-D Tensor, epsilon<=1e-03, momentum=0.1, data_format='NLC'
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
                    [
                        [0.6964692, 0.28613934, 0.22685145],
                        [0.5513148, 0.71946895, 0.42310646],
                        [0.9807642, 0.6848297, 0.4809319],
                    ],
                    [
                        [0.39211753, 0.343178, 0.7290497],
                        [0.43857226, 0.0596779, 0.39804426],
                        [0.7379954, 0.18249173, 0.17545176],
                    ],
                ]
            ).astype(t)
            x = paddle.to_tensor(x_data)
            batch_norm = paddle.nn.InstanceNorm1D(num_features=3, epsilon=1e-05, momentum=0.1, data_format="NLC")
            batch_norm_out = batch_norm(x)
            res = np.array(
                [
                    [
                        [1.4045197, -0.5603124, -0.8442078],
                        [-0.10968472, 1.2754805, -1.165797],
                        [1.2924801, -0.1494856, -1.1429948],
                    ],
                    [
                        [-0.55954623, -0.84480286, 1.4043493],
                        [0.82289475, -1.4072454, 0.5843504],
                        [1.4140275, -0.69365853, -0.7203695],
                    ],
                ]
            )
            compare(res, batch_norm_out.numpy())
            paddle.enable_static()


@pytest.mark.api_nn_InstanceNorm1D_parameters
def test_dygraph7():
    """
    input=3-D Tensor, epsilon<=1e-03, momentum=0.1, weight_attr=False, bias_attr=False
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
                    [[0.6964692, 0.28613934, 0.22685145], [0.5513148, 0.71946895, 0.42310646]],
                    [[0.9807642, 0.6848297, 0.4809319], [0.39211753, 0.343178, 0.7290497]],
                ]
            ).astype(t)
            x = paddle.to_tensor(x_data)
            batch_norm = paddle.nn.InstanceNorm1D(
                num_features=2, epsilon=1e-05, momentum=0.1, weight_attr=False, bias_attr=False
            )
            batch_norm_out = batch_norm(x)
            res = np.array(
                [
                    [[1.4045197, -0.5603124, -0.8442078], [-0.10968472, 1.2754805, -1.165797]],
                    [[1.2924801, -0.1494856, -1.1429948], [-0.55954623, -0.84480286, 1.4043493]],
                ]
            )
            compare(res, batch_norm_out.numpy())
            paddle.enable_static()


@pytest.mark.api_nn_InstanceNorm1D_parameters
def test_static1():
    """
    input=3-D Tensor
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
                    [[0.6964692, 0.28613934, 0.22685145], [0.5513148, 0.71946895, 0.42310646]],
                    [[0.9807642, 0.6848297, 0.4809319], [0.39211753, 0.343178, 0.7290497]],
                ]
            ).astype(t)
            feed = {"x_data": x_data}
            with paddle.utils.unique_name.guard():
                with static.program_guard(main_program=main_program, startup_program=startup_program):
                    x_data = static.data(name="x_data", shape=[2, 2, 3], dtype=t)
                    exe = static.Executor(place)
                    output = paddle.nn.InstanceNorm1D(2)(x_data)
                    exe.run(startup_program)
                    res = exe.run(main_program, feed=feed, fetch_list=[output])
                    expect = np.array(
                        [
                            [[1.4045197, -0.5603124, -0.8442078], [-0.10968472, 1.2754805, -1.165797]],
                            [[1.2924801, -0.1494856, -1.1429948], [-0.55954623, -0.84480286, 1.4043493]],
                        ]
                    )
                    compare(res[0], expect)


@pytest.mark.api_nn_InstanceNorm1D_parameters
def test_static2():
    """
    input=2-D Tensor, momentum=0.1
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
                    [[0.6964692, 0.28613934, 0.22685145], [0.5513148, 0.71946895, 0.42310646]],
                    [[0.9807642, 0.6848297, 0.4809319], [0.39211753, 0.343178, 0.7290497]],
                ]
            ).astype(t)
            feed = {"x_data": x_data}
            with paddle.utils.unique_name.guard():
                with static.program_guard(main_program=main_program, startup_program=startup_program):
                    x_data = static.data(name="x_data", shape=[2, 2, 3], dtype=t)
                    exe = static.Executor(place)
                    output = paddle.nn.InstanceNorm1D(num_features=2, momentum=0.1)(x_data)
                    exe.run(startup_program)
                    res = exe.run(main_program, feed=feed, fetch_list=[output])
                    expect = np.array(
                        [
                            [[1.4045197, -0.5603124, -0.8442078], [-0.10968472, 1.2754805, -1.165797]],
                            [[1.2924801, -0.1494856, -1.1429948], [-0.55954623, -0.84480286, 1.4043493]],
                        ]
                    )
                    compare(res[0], expect)


@pytest.mark.api_nn_InstanceNorm1D_parameters
def test_static3():
    """
    input=2-D Tensor, momentum=0.1, epsilon<=1e-03
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
                    [[0.6964692, 0.28613934, 0.22685145], [0.5513148, 0.71946895, 0.42310646]],
                    [[0.9807642, 0.6848297, 0.4809319], [0.39211753, 0.343178, 0.7290497]],
                ]
            ).astype(t)
            feed = {"x_data": x_data}
            with paddle.utils.unique_name.guard():
                with static.program_guard(main_program=main_program, startup_program=startup_program):
                    x_data = static.data(name="x_data", shape=[2, 2, 3], dtype=t)
                    exe = static.Executor(place)
                    output = paddle.nn.InstanceNorm1D(num_features=2, momentum=0.1, epsilon=1e-05)(x_data)
                    exe.run(startup_program)
                    res = exe.run(main_program, feed=feed, fetch_list=[output])
                    expect = np.array(
                        [
                            [[1.4045197, -0.5603124, -0.8442078], [-0.10968472, 1.2754805, -1.165797]],
                            [[1.2924801, -0.1494856, -1.1429948], [-0.55954623, -0.84480286, 1.4043493]],
                        ]
                    )
                    compare(res[0], expect)


@pytest.mark.api_nn_InstanceNorm1D_parameters
def test_static4():
    """
    input=2-D Tensor, momentum=0.1, epsilon<=1e-03, weight_attr=False, bias_attr=False
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
                    [[0.6964692, 0.28613934, 0.22685145], [0.5513148, 0.71946895, 0.42310646]],
                    [[0.9807642, 0.6848297, 0.4809319], [0.39211753, 0.343178, 0.7290497]],
                ]
            ).astype(t)
            feed = {"x_data": x_data}
            with paddle.utils.unique_name.guard():
                with static.program_guard(main_program=main_program, startup_program=startup_program):
                    w_param_attrs = paddle.ParamAttr(
                        name="w_weight", learning_rate=0.5, regularizer=paddle.regularizer.L2Decay(1.0), trainable=True
                    )
                    b_param_attrs = paddle.ParamAttr(
                        name="b_weight", learning_rate=0.5, regularizer=paddle.regularizer.L2Decay(1.0), trainable=True
                    )
                    x_data = static.data(name="x_data", shape=[2, 2, 3], dtype=t)
                    exe = static.Executor(place)
                    output = paddle.nn.InstanceNorm1D(
                        num_features=2, momentum=0.1, epsilon=1e-05, weight_attr=w_param_attrs, bias_attr=b_param_attrs
                    )(x_data)
                    exe.run(startup_program)
                    res = exe.run(main_program, feed=feed, fetch_list=[output])
                    expect = np.array(
                        [
                            [[1.4045197, -0.5603124, -0.8442078], [-0.10968472, 1.2754805, -1.165797]],
                            [[1.2924801, -0.1494856, -1.1429948], [-0.55954623, -0.84480286, 1.4043493]],
                        ]
                    )
                    compare(res[0], expect)


@pytest.mark.api_nn_InstanceNorm1D_parameters
def test_static5():
    """
    input=2-D Tensor, momentum=0.1, epsilon<=1e-03, data_format='NCL'
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
                    [[0.6964692, 0.28613934, 0.22685145], [0.5513148, 0.71946895, 0.42310646]],
                    [[0.9807642, 0.6848297, 0.4809319], [0.39211753, 0.343178, 0.7290497]],
                ]
            ).astype(t)
            feed = {"x_data": x_data}
            with paddle.utils.unique_name.guard():
                with static.program_guard(main_program=main_program, startup_program=startup_program):
                    x_data = static.data(name="x_data", shape=[2, 2, 3], dtype=t)
                    exe = static.Executor(place)
                    output = paddle.nn.InstanceNorm1D(num_features=2, momentum=0.1, epsilon=1e-05, data_format="NCL")(
                        x_data
                    )
                    exe.run(startup_program)
                    res = exe.run(main_program, feed=feed, fetch_list=[output])
                    expect = np.array(
                        [
                            [[1.4045197, -0.5603124, -0.8442078], [-0.10968472, 1.2754805, -1.165797]],
                            [[1.2924801, -0.1494856, -1.1429948], [-0.55954623, -0.84480286, 1.4043493]],
                        ]
                    )
                    compare(res[0], expect)


@pytest.mark.api_nn_InstanceNorm1D_parameters
def test_static6():
    """
    input=2-D Tensor, momentum=0.1, epsilon<=1e-03, data_format='NLC'
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
                    [
                        [0.6964692, 0.28613934, 0.22685145],
                        [0.5513148, 0.71946895, 0.42310646],
                        [0.9807642, 0.6848297, 0.4809319],
                    ],
                    [
                        [0.39211753, 0.343178, 0.7290497],
                        [0.43857226, 0.0596779, 0.39804426],
                        [0.7379954, 0.18249173, 0.17545176],
                    ],
                ]
            ).astype(t)
            feed = {"x_data": x_data}
            with paddle.utils.unique_name.guard():
                with static.program_guard(main_program=main_program, startup_program=startup_program):
                    x_data = static.data(name="x_data", shape=[2, 3, 3], dtype=t)
                    exe = static.Executor(place)
                    output = paddle.nn.InstanceNorm1D(num_features=3, momentum=0.1, epsilon=1e-05, data_format="NLC")(
                        x_data
                    )
                    exe.run(startup_program)
                    res = exe.run(main_program, feed=feed, fetch_list=[output])
                    expect = np.array(
                        [
                            [
                                [1.4045197, -0.5603124, -0.8442078],
                                [-0.10968472, 1.2754805, -1.165797],
                                [1.2924801, -0.1494856, -1.1429948],
                            ],
                            [
                                [-0.55954623, -0.84480286, 1.4043493],
                                [0.82289475, -1.4072454, 0.5843504],
                                [1.4140275, -0.69365853, -0.7203695],
                            ],
                        ]
                    )
                    compare(res[0], expect)
