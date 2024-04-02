#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_batchnorm3D
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


@pytest.mark.api_nn_BatchNorm3D_parameters
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
                    [
                        [
                            [[0.6964692, 0.28613934, 0.22685145], [0.5513148, 0.71946895, 0.42310646]],
                            [[0.9807642, 0.6848297, 0.4809319], [0.39211753, 0.343178, 0.7290497]],
                        ]
                    ],
                    [
                        [
                            [[0.43857226, 0.0596779, 0.39804426], [0.7379954, 0.18249173, 0.17545176]],
                            [[0.53155136, 0.53182757, 0.63440096], [0.8494318, 0.7244553, 0.6110235]],
                        ]
                    ],
                ]
            ).astype(t)
            x = paddle.to_tensor(x_data)
            batch_norm = paddle.nn.BatchNorm3D(1)
            batch_norm_out = batch_norm(x)
            res = np.array(
                [
                    [
                        [
                            [[0.79312336, -1.0123329, -1.2732003], [0.15444219, 0.8943226, -0.40967593]],
                            [[2.044025, 0.7419095, -0.15524326], [-0.5460276, -0.7613621, 0.93647796]],
                        ]
                    ],
                    [
                        [
                            [[-0.34162623, -2.008766, -0.5199499], [0.97583914, -1.4683836, -1.4993596]],
                            [[0.06748295, 0.06869826, 0.52002245], [1.4661608, 0.91626257, 0.41716126]],
                        ]
                    ],
                ]
            )
            compare(res, batch_norm_out.numpy())
            paddle.enable_static()


@pytest.mark.api_nn_BatchNorm3D_parameters
def test_dygraph2():
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

            paddle.disable_static(place)
            x_data = np.array(
                [
                    [
                        [
                            [[0.6964692, 0.28613934, 0.22685145], [0.5513148, 0.71946895, 0.42310646]],
                            [[0.9807642, 0.6848297, 0.4809319], [0.39211753, 0.343178, 0.7290497]],
                        ]
                    ],
                    [
                        [
                            [[0.43857226, 0.0596779, 0.39804426], [0.7379954, 0.18249173, 0.17545176]],
                            [[0.53155136, 0.53182757, 0.63440096], [0.8494318, 0.7244553, 0.6110235]],
                        ]
                    ],
                ]
            ).astype(t)
            x = paddle.to_tensor(x_data)
            batch_norm = paddle.nn.BatchNorm3D(num_features=1, momentum=0.1)
            batch_norm_out = batch_norm(x)
            res = np.array(
                [
                    [
                        [
                            [[0.79312336, -1.0123329, -1.2732003], [0.15444219, 0.8943226, -0.40967593]],
                            [[2.044025, 0.7419095, -0.15524326], [-0.5460276, -0.7613621, 0.93647796]],
                        ]
                    ],
                    [
                        [
                            [[-0.34162623, -2.008766, -0.5199499], [0.97583914, -1.4683836, -1.4993596]],
                            [[0.06748295, 0.06869826, 0.52002245], [1.4661608, 0.91626257, 0.41716126]],
                        ]
                    ],
                ]
            )
            compare(res, batch_norm_out.numpy())
            paddle.enable_static()


@pytest.mark.api_nn_BatchNorm3D_parameters
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
                    [
                        [
                            [[0.6964692, 0.28613934, 0.22685145], [0.5513148, 0.71946895, 0.42310646]],
                            [[0.9807642, 0.6848297, 0.4809319], [0.39211753, 0.343178, 0.7290497]],
                        ]
                    ],
                    [
                        [
                            [[0.43857226, 0.0596779, 0.39804426], [0.7379954, 0.18249173, 0.17545176]],
                            [[0.53155136, 0.53182757, 0.63440096], [0.8494318, 0.7244553, 0.6110235]],
                        ]
                    ],
                ]
            ).astype(t)
            x = paddle.to_tensor(x_data)
            batch_norm = paddle.nn.BatchNorm3D(num_features=1, momentum=0.1, epsilon=1e-5)
            batch_norm_out = batch_norm(x)
            res = np.array(
                [
                    [
                        [
                            [[0.79312336, -1.0123329, -1.2732003], [0.15444219, 0.8943226, -0.40967593]],
                            [[2.044025, 0.7419095, -0.15524326], [-0.5460276, -0.7613621, 0.93647796]],
                        ]
                    ],
                    [
                        [
                            [[-0.34162623, -2.008766, -0.5199499], [0.97583914, -1.4683836, -1.4993596]],
                            [[0.06748295, 0.06869826, 0.52002245], [1.4661608, 0.91626257, 0.41716126]],
                        ]
                    ],
                ]
            )
            compare(res, batch_norm_out.numpy())
            paddle.enable_static()


@pytest.mark.api_nn_BatchNorm3D_parameters
def test_dygraph4():
    """
    input=4-D Tensor, momentum=0.1, epsilon<=1e-3, weight_attr=False, bias_attr=b
    :return:
    """
    for place in places:
        for t in types:
            if t == np.float64:
                paddle.set_default_dtype("float64")
            elif t == np.float32:
                paddle.set_default_dtype("float32")

            paddle.disable_static(place)
            # w_param_attrs = paddle.ParamAttr(name="w1_weight",
            #                                 learning_rate=0.5,
            #                                 regularizer=paddle.regularizer.L2Decay(1.0),
            #                                 trainable=True)
            b_param_attrs = paddle.ParamAttr(
                name="b1_weight", learning_rate=0.5, regularizer=paddle.regularizer.L2Decay(1.0), trainable=True
            )
            x_data = np.array(
                [
                    [
                        [
                            [[0.6964692, 0.28613934, 0.22685145], [0.5513148, 0.71946895, 0.42310646]],
                            [[0.9807642, 0.6848297, 0.4809319], [0.39211753, 0.343178, 0.7290497]],
                        ]
                    ],
                    [
                        [
                            [[0.43857226, 0.0596779, 0.39804426], [0.7379954, 0.18249173, 0.17545176]],
                            [[0.53155136, 0.53182757, 0.63440096], [0.8494318, 0.7244553, 0.6110235]],
                        ]
                    ],
                ]
            ).astype(t)
            x = paddle.to_tensor(x_data)
            batch_norm = paddle.nn.BatchNorm3D(
                num_features=1, momentum=0.1, epsilon=1e-5, weight_attr=False, bias_attr=b_param_attrs
            )
            batch_norm_out = batch_norm(x)
            res = np.array(
                [
                    [
                        [
                            [[0.79312336, -1.0123329, -1.2732003], [0.15444219, 0.8943226, -0.40967593]],
                            [[2.044025, 0.7419095, -0.15524326], [-0.5460276, -0.7613621, 0.93647796]],
                        ]
                    ],
                    [
                        [
                            [[-0.34162623, -2.008766, -0.5199499], [0.97583914, -1.4683836, -1.4993596]],
                            [[0.06748295, 0.06869826, 0.52002245], [1.4661608, 0.91626257, 0.41716126]],
                        ]
                    ],
                ]
            )
            compare(res, batch_norm_out.numpy())
            paddle.enable_static()


@pytest.mark.api_nn_BatchNorm3D_parameters
def test_dygraph5():
    """
    input=4-D Tensor, momentum=0.1, epsilon<=1e-3, data_format='NCDHW'
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
                        [
                            [[0.6964692, 0.28613934, 0.22685145], [0.5513148, 0.71946895, 0.42310646]],
                            [[0.9807642, 0.6848297, 0.4809319], [0.39211753, 0.343178, 0.7290497]],
                        ]
                    ],
                    [
                        [
                            [[0.43857226, 0.0596779, 0.39804426], [0.7379954, 0.18249173, 0.17545176]],
                            [[0.53155136, 0.53182757, 0.63440096], [0.8494318, 0.7244553, 0.6110235]],
                        ]
                    ],
                ]
            ).astype(t)
            x = paddle.to_tensor(x_data)
            batch_norm = paddle.nn.BatchNorm3D(num_features=1, momentum=0.1, epsilon=1e-5, data_format="NCDHW")
            batch_norm_out = batch_norm(x)
            res = np.array(
                [
                    [
                        [
                            [[0.79312336, -1.0123329, -1.2732003], [0.15444219, 0.8943226, -0.40967593]],
                            [[2.044025, 0.7419095, -0.15524326], [-0.5460276, -0.7613621, 0.93647796]],
                        ]
                    ],
                    [
                        [
                            [[-0.34162623, -2.008766, -0.5199499], [0.97583914, -1.4683836, -1.4993596]],
                            [[0.06748295, 0.06869826, 0.52002245], [1.4661608, 0.91626257, 0.41716126]],
                        ]
                    ],
                ]
            )
            compare(res, batch_norm_out.numpy())
            paddle.enable_static()


@pytest.mark.api_nn_BatchNorm3D_parameters
def test_dygraph6():
    """
    input=4-D Tensor, momentum=0.1, epsilon<=1e-3, weight_attr=w, bias_attr=b
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
                    [
                        [
                            [[0.6964692, 0.28613934, 0.22685145], [0.5513148, 0.71946895, 0.42310646]],
                            [[0.9807642, 0.6848297, 0.4809319], [0.39211753, 0.343178, 0.7290497]],
                        ]
                    ],
                    [
                        [
                            [[0.43857226, 0.0596779, 0.39804426], [0.7379954, 0.18249173, 0.17545176]],
                            [[0.53155136, 0.53182757, 0.63440096], [0.8494318, 0.7244553, 0.6110235]],
                        ]
                    ],
                ]
            ).astype(t)
            x = paddle.to_tensor(x_data)
            batch_norm = paddle.nn.BatchNorm3D(
                num_features=1, momentum=0.1, epsilon=1e-5, weight_attr=w_param_attrs, bias_attr=b_param_attrs
            )
            batch_norm_out = batch_norm(x)
            res = np.array(
                [
                    [
                        [
                            [[0.79312336, -1.0123329, -1.2732003], [0.15444219, 0.8943226, -0.40967593]],
                            [[2.044025, 0.7419095, -0.15524326], [-0.5460276, -0.7613621, 0.93647796]],
                        ]
                    ],
                    [
                        [
                            [[-0.34162623, -2.008766, -0.5199499], [0.97583914, -1.4683836, -1.4993596]],
                            [[0.06748295, 0.06869826, 0.52002245], [1.4661608, 0.91626257, 0.41716126]],
                        ]
                    ],
                ]
            )
            compare(res, batch_norm_out.numpy())
            paddle.enable_static()


@pytest.mark.api_nn_BatchNorm3D_parameters
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
                    [
                        [
                            [[0.6964692, 0.28613934, 0.22685145], [0.5513148, 0.71946895, 0.42310646]],
                            [[0.9807642, 0.6848297, 0.4809319], [0.39211753, 0.343178, 0.7290497]],
                        ]
                    ],
                    [
                        [
                            [[0.43857226, 0.0596779, 0.39804426], [0.7379954, 0.18249173, 0.17545176]],
                            [[0.53155136, 0.53182757, 0.63440096], [0.8494318, 0.7244553, 0.6110235]],
                        ]
                    ],
                ]
            ).astype(t)
            feed = {"x_data": x_data}
            with paddle.utils.unique_name.guard():
                with static.program_guard(main_program=main_program, startup_program=startup_program):
                    x_data = static.data(name="x_data", shape=[2, 1, 2, 2, 3], dtype=t)
                    exe = static.Executor(place)
                    output = paddle.nn.BatchNorm3D(1)(x_data)
                    exe.run(startup_program)
                    res = exe.run(main_program, feed=feed, fetch_list=[output])
                    expect = np.array(
                        [
                            [
                                [
                                    [[0.79312336, -1.0123329, -1.2732003], [0.15444219, 0.8943226, -0.40967593]],
                                    [[2.044025, 0.7419095, -0.15524326], [-0.5460276, -0.7613621, 0.93647796]],
                                ]
                            ],
                            [
                                [
                                    [[-0.34162623, -2.008766, -0.5199499], [0.97583914, -1.4683836, -1.4993596]],
                                    [[0.06748295, 0.06869826, 0.52002245], [1.4661608, 0.91626257, 0.41716126]],
                                ]
                            ],
                        ]
                    )
                    compare(res[0], expect)


@pytest.mark.api_nn_BatchNorm3D_parameters
def test_static2():
    """
    input=4-D Tensor, momentum=0.1,
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
                        [
                            [[0.6964692, 0.28613934, 0.22685145], [0.5513148, 0.71946895, 0.42310646]],
                            [[0.9807642, 0.6848297, 0.4809319], [0.39211753, 0.343178, 0.7290497]],
                        ]
                    ],
                    [
                        [
                            [[0.43857226, 0.0596779, 0.39804426], [0.7379954, 0.18249173, 0.17545176]],
                            [[0.53155136, 0.53182757, 0.63440096], [0.8494318, 0.7244553, 0.6110235]],
                        ]
                    ],
                ]
            ).astype(t)
            feed = {"x_data": x_data}
            with paddle.utils.unique_name.guard():
                with static.program_guard(main_program=main_program, startup_program=startup_program):
                    x_data = static.data(name="x_data", shape=[2, 1, 2, 2, 3], dtype=t)
                    exe = static.Executor(place)
                    output = paddle.nn.BatchNorm3D(num_features=1, momentum=0.1)(x_data)
                    exe.run(startup_program)
                    res = exe.run(main_program, feed=feed, fetch_list=[output])
                    expect = np.array(
                        [
                            [
                                [
                                    [[0.79312336, -1.0123329, -1.2732003], [0.15444219, 0.8943226, -0.40967593]],
                                    [[2.044025, 0.7419095, -0.15524326], [-0.5460276, -0.7613621, 0.93647796]],
                                ]
                            ],
                            [
                                [
                                    [[-0.34162623, -2.008766, -0.5199499], [0.97583914, -1.4683836, -1.4993596]],
                                    [[0.06748295, 0.06869826, 0.52002245], [1.4661608, 0.91626257, 0.41716126]],
                                ]
                            ],
                        ]
                    )
                    compare(res[0], expect)


@pytest.mark.api_nn_BatchNorm3D_parameters
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
                    [
                        [
                            [[0.6964692, 0.28613934, 0.22685145], [0.5513148, 0.71946895, 0.42310646]],
                            [[0.9807642, 0.6848297, 0.4809319], [0.39211753, 0.343178, 0.7290497]],
                        ]
                    ],
                    [
                        [
                            [[0.43857226, 0.0596779, 0.39804426], [0.7379954, 0.18249173, 0.17545176]],
                            [[0.53155136, 0.53182757, 0.63440096], [0.8494318, 0.7244553, 0.6110235]],
                        ]
                    ],
                ]
            ).astype(t)
            feed = {"x_data": x_data}
            with paddle.utils.unique_name.guard():
                with static.program_guard(main_program=main_program, startup_program=startup_program):
                    x_data = static.data(name="x_data", shape=[2, 1, 2, 2, 3], dtype=t)
                    exe = static.Executor(place)
                    output = paddle.nn.BatchNorm3D(num_features=1, momentum=0.1, epsilon=1e-5)(x_data)
                    exe.run(startup_program)
                    res = exe.run(main_program, feed=feed, fetch_list=[output])
                    expect = np.array(
                        [
                            [
                                [
                                    [[0.79312336, -1.0123329, -1.2732003], [0.15444219, 0.8943226, -0.40967593]],
                                    [[2.044025, 0.7419095, -0.15524326], [-0.5460276, -0.7613621, 0.93647796]],
                                ]
                            ],
                            [
                                [
                                    [[-0.34162623, -2.008766, -0.5199499], [0.97583914, -1.4683836, -1.4993596]],
                                    [[0.06748295, 0.06869826, 0.52002245], [1.4661608, 0.91626257, 0.41716126]],
                                ]
                            ],
                        ]
                    )
                    compare(res[0], expect)


@pytest.mark.api_nn_BatchNorm3D_parameters
def test_static4():
    """
    input=4-D Tensor, momentum=0.1, epsilon<=1e-3, data_format='NCDHW'
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
                        [
                            [[0.6964692, 0.28613934, 0.22685145], [0.5513148, 0.71946895, 0.42310646]],
                            [[0.9807642, 0.6848297, 0.4809319], [0.39211753, 0.343178, 0.7290497]],
                        ]
                    ],
                    [
                        [
                            [[0.43857226, 0.0596779, 0.39804426], [0.7379954, 0.18249173, 0.17545176]],
                            [[0.53155136, 0.53182757, 0.63440096], [0.8494318, 0.7244553, 0.6110235]],
                        ]
                    ],
                ]
            ).astype(t)
            feed = {"x_data": x_data}
            with paddle.utils.unique_name.guard():
                with static.program_guard(main_program=main_program, startup_program=startup_program):
                    x_data = static.data(name="x_data", shape=[2, 1, 2, 2, 3], dtype=t)
                    exe = static.Executor(place)
                    output = paddle.nn.BatchNorm3D(num_features=1, momentum=0.1, epsilon=1e-5, data_format="NCDHW")(
                        x_data
                    )
                    exe.run(startup_program)
                    res = exe.run(main_program, feed=feed, fetch_list=[output])
                    expect = np.array(
                        [
                            [
                                [
                                    [[0.79312336, -1.0123329, -1.2732003], [0.15444219, 0.8943226, -0.40967593]],
                                    [[2.044025, 0.7419095, -0.15524326], [-0.5460276, -0.7613621, 0.93647796]],
                                ]
                            ],
                            [
                                [
                                    [[-0.34162623, -2.008766, -0.5199499], [0.97583914, -1.4683836, -1.4993596]],
                                    [[0.06748295, 0.06869826, 0.52002245], [1.4661608, 0.91626257, 0.41716126]],
                                ]
                            ],
                        ]
                    )
                    compare(res[0], expect)


@pytest.mark.api_nn_BatchNorm3D_parameters
def test_static5():
    """
    input=4-D Tensor, momentum=0.1, epsilon<=1e-3, data_format='NCDHW', weight_attr=w, bias_attr=b
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
                        [
                            [[0.6964692, 0.28613934, 0.22685145], [0.5513148, 0.71946895, 0.42310646]],
                            [[0.9807642, 0.6848297, 0.4809319], [0.39211753, 0.343178, 0.7290497]],
                        ]
                    ],
                    [
                        [
                            [[0.43857226, 0.0596779, 0.39804426], [0.7379954, 0.18249173, 0.17545176]],
                            [[0.53155136, 0.53182757, 0.63440096], [0.8494318, 0.7244553, 0.6110235]],
                        ]
                    ],
                ]
            ).astype(t)
            feed = {"x_data": x_data}
            with paddle.utils.unique_name.guard():
                with static.program_guard(main_program=main_program, startup_program=startup_program):
                    x_data = static.data(name="x_data", shape=[2, 1, 2, 2, 3], dtype=t)
                    w_param_attrs = paddle.ParamAttr(
                        name="w1_weight", learning_rate=0.5, regularizer=paddle.regularizer.L2Decay(1.0), trainable=True
                    )
                    b_param_attrs = paddle.ParamAttr(
                        name="b1_weight", learning_rate=0.5, regularizer=paddle.regularizer.L2Decay(1.0), trainable=True
                    )
                    exe = static.Executor(place)
                    output = paddle.nn.BatchNorm3D(
                        num_features=1,
                        momentum=0.1,
                        epsilon=1e-5,
                        data_format="NCDHW",
                        weight_attr=w_param_attrs,
                        bias_attr=b_param_attrs,
                    )(x_data)
                    exe.run(startup_program)
                    res = exe.run(main_program, feed=feed, fetch_list=[output])
                    expect = np.array(
                        [
                            [
                                [
                                    [[0.79312336, -1.0123329, -1.2732003], [0.15444219, 0.8943226, -0.40967593]],
                                    [[2.044025, 0.7419095, -0.15524326], [-0.5460276, -0.7613621, 0.93647796]],
                                ]
                            ],
                            [
                                [
                                    [[-0.34162623, -2.008766, -0.5199499], [0.97583914, -1.4683836, -1.4993596]],
                                    [[0.06748295, 0.06869826, 0.52002245], [1.4661608, 0.91626257, 0.41716126]],
                                ]
                            ],
                        ]
                    )
                    compare(res[0], expect)


@pytest.mark.api_nn_BatchNorm3D_parameters
def test_static6():
    """
    input=4-D Tensor, momentum=0.1, epsilon<=1e-3, data_format='NCDHW', weight_attr=False, bias_attr=b
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
                        [
                            [[0.6964692, 0.28613934, 0.22685145], [0.5513148, 0.71946895, 0.42310646]],
                            [[0.9807642, 0.6848297, 0.4809319], [0.39211753, 0.343178, 0.7290497]],
                        ]
                    ],
                    [
                        [
                            [[0.43857226, 0.0596779, 0.39804426], [0.7379954, 0.18249173, 0.17545176]],
                            [[0.53155136, 0.53182757, 0.63440096], [0.8494318, 0.7244553, 0.6110235]],
                        ]
                    ],
                ]
            ).astype(t)
            feed = {"x_data": x_data}
            with paddle.utils.unique_name.guard():
                with static.program_guard(main_program=main_program, startup_program=startup_program):
                    x_data = static.data(name="x_data", shape=[2, 1, 2, 2, 3], dtype=t)
                    # w_param_attrs = paddle.ParamAttr(name="w1_weight",
                    #                                 learning_rate=0.5,
                    #                                 regularizer=paddle.regularizer.L2Decay(1.0),
                    #                                 trainable=True)
                    b_param_attrs = paddle.ParamAttr(
                        name="b1_weight", learning_rate=0.5, regularizer=paddle.regularizer.L2Decay(1.0), trainable=True
                    )
                    exe = static.Executor(place)
                    output = paddle.nn.BatchNorm3D(
                        num_features=1,
                        momentum=0.1,
                        epsilon=1e-5,
                        data_format="NCDHW",
                        weight_attr=False,
                        bias_attr=b_param_attrs,
                    )(x_data)
                    exe.run(startup_program)
                    res = exe.run(main_program, feed=feed, fetch_list=[output])
                    expect = np.array(
                        [
                            [
                                [
                                    [[0.79312336, -1.0123329, -1.2732003], [0.15444219, 0.8943226, -0.40967593]],
                                    [[2.044025, 0.7419095, -0.15524326], [-0.5460276, -0.7613621, 0.93647796]],
                                ]
                            ],
                            [
                                [
                                    [[-0.34162623, -2.008766, -0.5199499], [0.97583914, -1.4683836, -1.4993596]],
                                    [[0.06748295, 0.06869826, 0.52002245], [1.4661608, 0.91626257, 0.41716126]],
                                ]
                            ],
                        ]
                    )
                    compare(res[0], expect)
