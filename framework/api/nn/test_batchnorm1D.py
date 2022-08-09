#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_batchnorm1D
"""
import paddle
import pytest
import numpy as np
from apibase import compare

seed = 33
types = [np.float32, np.float64]
if paddle.device.is_compiled_with_cuda() is True:
    places = [paddle.CPUPlace(), paddle.CUDAPlace(0)]
else:
    places = [paddle.CPUPlace()]


@pytest.mark.api_nn_BatchNorm1D_parameters
def test_dygraph1():
    """
    input=2-D Tensor
    :return:
    """
    for place in places:
        for t in types:
            if t == np.float64:
                paddle.set_default_dtype("float64")
            elif t == np.float32:
                paddle.set_default_dtype("float32")

            paddle.disable_static(place)
            x_data = np.array([[0.6964692], [0.28613934]]).astype(t)
            x = paddle.to_tensor(x_data)
            batch_norm = paddle.nn.BatchNorm1D(1)
            batch_norm_out = batch_norm(x)
            res = np.array([[0.99988127], [-0.99988127]])
            compare(res, batch_norm_out.numpy())
            paddle.enable_static()


@pytest.mark.api_nn_BatchNorm1D_parameters
def test_dygraph2():
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
            x_data = np.array([[[0.6964692, 0.28613934, 0.22685145]], [[0.5513148, 0.71946895, 0.42310646]]]).astype(t)
            x = paddle.to_tensor(x_data)
            batch_norm = paddle.nn.BatchNorm1D(1)
            batch_norm_out = batch_norm(x)

            res1 = np.array([[1.1251787, -1.0467086, -1.3605211]])
            res2 = np.array([[0.35687235, 1.2469171, -0.32173783]])
            compare(res1, batch_norm_out.numpy()[0])
            compare(res2, batch_norm_out.numpy()[1])
            paddle.enable_static()


@pytest.mark.api_nn_BatchNorm1D_parameters
def test_dygraph3():
    """
    input=2-D Tensor, momentum=0.1.
    :return:
    """
    for place in places:
        for t in types:
            if t == np.float64:
                paddle.set_default_dtype("float64")
            elif t == np.float32:
                paddle.set_default_dtype("float32")

            paddle.disable_static(place)
            x_data = np.array([[0.6964692], [0.28613934]]).astype(t)
            x = paddle.to_tensor(x_data)
            batch_norm = paddle.nn.BatchNorm1D(num_features=1, momentum=0.1)
            batch_norm_out = batch_norm(x)
            res = np.array([[0.99988127], [-0.99988127]])
            compare(res, batch_norm_out.numpy())
            paddle.enable_static()


@pytest.mark.api_nn_BatchNorm1D_parameters
def test_dygraph4():
    """
    input=3-D Tensor, momentum=0.1.
    :return:
    """
    for place in places:
        for t in types:
            if t == np.float64:
                paddle.set_default_dtype("float64")
            elif t == np.float32:
                paddle.set_default_dtype("float32")

            paddle.disable_static(place)
            x_data = np.array([[[0.6964692, 0.28613934, 0.22685145]], [[0.5513148, 0.71946895, 0.42310646]]]).astype(t)
            x = paddle.to_tensor(x_data)
            batch_norm = paddle.nn.BatchNorm1D(num_features=1, momentum=0.1)
            batch_norm_out = batch_norm(x)

            res1 = np.array([[1.1251787, -1.0467086, -1.3605211]])
            res2 = np.array([[0.35687235, 1.2469171, -0.32173783]])
            compare(res1, batch_norm_out.numpy()[0])
            compare(res2, batch_norm_out.numpy()[1])
            paddle.enable_static()


@pytest.mark.api_nn_BatchNorm1D_parameters
def test_dygraph5():
    """
    input=2-D Tensor, momentum=0.1, epsilon<=1e-3
    :return:
    """
    for place in places:
        for t in types:
            if t == np.float64:
                paddle.set_default_dtype("float64")
            elif t == np.float32:
                paddle.set_default_dtype("float32")

            paddle.disable_static(place)
            x_data = np.array([[0.6964692], [0.28613934]]).astype(t)
            x = paddle.to_tensor(x_data)
            batch_norm = paddle.nn.BatchNorm1D(num_features=1, momentum=0.1, epsilon=1e-5)
            batch_norm_out = batch_norm(x)
            res = np.array([[0.99988127], [-0.99988127]])
            compare(res, batch_norm_out.numpy())
            paddle.enable_static()


@pytest.mark.api_nn_BatchNorm1D_parameters
def test_dygraph6():
    """
    input=3-D Tensor, momentum=0.1, epsilon<=1e-3
    :return:
    """
    for place in places:
        for t in types:
            if t == np.float64:
                paddle.set_default_dtype("float64")
            elif t == np.float32:
                paddle.set_default_dtype("float32")

            paddle.disable_static(place)
            x_data = np.array([[[0.6964692, 0.28613934, 0.22685145]], [[0.5513148, 0.71946895, 0.42310646]]]).astype(t)
            x = paddle.to_tensor(x_data)
            batch_norm = paddle.nn.BatchNorm1D(num_features=1, momentum=0.1, epsilon=1e-5)
            batch_norm_out = batch_norm(x)

            res1 = np.array([[1.1251787, -1.0467086, -1.3605211]])
            res2 = np.array([[0.35687235, 1.2469171, -0.32173783]])
            compare(res1, batch_norm_out.numpy()[0])
            compare(res2, batch_norm_out.numpy()[1])
            paddle.enable_static()


@pytest.mark.api_nn_BatchNorm1D_parameters
def test_dygraph7():
    """
    input=2-D Tensor, momentum=0.1, epsilon<=1e-3, weight_attr=False, bias_attr=False, data_format='NCL'
    :return:
    """
    for place in places:
        for t in types:
            if t == np.float64:
                paddle.set_default_dtype("float64")
            elif t == np.float32:
                paddle.set_default_dtype("float32")

            w_param_attrs = paddle.ParamAttr(
                name="w1_weight", learning_rate=0.5, regularizer=paddle.regularizer.L2Decay(1.0), trainable=True
            )
            b_param_attrs = paddle.ParamAttr(
                name="b1_weight", learning_rate=0.5, regularizer=paddle.regularizer.L2Decay(1.0), trainable=True
            )
            paddle.disable_static(place)
            x_data = np.array([[0.6964692], [0.28613934]]).astype(t)
            x = paddle.to_tensor(x_data)
            batch_norm = paddle.nn.BatchNorm1D(
                num_features=1,
                momentum=0.1,
                epsilon=1e-05,
                weight_attr=w_param_attrs,
                bias_attr=b_param_attrs,
                data_format="NCL",
            )
            batch_norm_out = batch_norm(x)
            res = np.array([[0.99988127], [-0.99988127]])
            compare(res, batch_norm_out.numpy())
            paddle.enable_static()


@pytest.mark.api_nn_BatchNorm1D_parameters
def test_dygraph8():
    """
    input=3-D Tensor, momentum=0.1, epsilon<=1e-3, weight_attr=False, bias_attr=attrs, data_format='NCL'
    :return:
    """
    for place in places:
        for t in types:
            if t == np.float64:
                paddle.set_default_dtype("float64")
            elif t == np.float32:
                paddle.set_default_dtype("float32")

            b_param_attrs = paddle.ParamAttr(
                name="b2_weight", learning_rate=0.5, regularizer=paddle.regularizer.L2Decay(1.0), trainable=True
            )
            paddle.disable_static(place)
            x_data = np.array([[[0.6964692, 0.28613934, 0.22685145]], [[0.5513148, 0.71946895, 0.42310646]]]).astype(t)
            x = paddle.to_tensor(x_data)
            batch_norm = paddle.nn.BatchNorm1D(
                num_features=1,
                momentum=0.1,
                epsilon=1e-5,
                weight_attr=False,
                bias_attr=b_param_attrs,
                data_format="NCL",
            )
            batch_norm_out = batch_norm(x)

            res1 = np.array([[1.1251787, -1.0467086, -1.3605211]])
            res2 = np.array([[0.35687235, 1.2469171, -0.32173783]])
            compare(res1, batch_norm_out.numpy()[0])
            compare(res2, batch_norm_out.numpy()[1])
            paddle.enable_static()


@pytest.mark.api_nn_BatchNorm1D_parameters
def test_static1():
    """
    input=2-D Tensor
    :return:
    """
    for place in places:
        for t in types:
            if t == np.float64:
                paddle.set_default_dtype("float64")
            elif t == np.float32:
                paddle.set_default_dtype("float32")

            main_program = paddle.static.Program()
            startup_program = paddle.static.Program()
            main_program.random_seed = seed
            startup_program.random_seed = seed
            x_data = np.array([[0.6964692], [0.28613934]]).astype(t)
            feed = {"x_data": x_data}
            with paddle.utils.unique_name.guard():
                with paddle.static.program_guard(main_program=main_program, startup_program=startup_program):
                    x_data = paddle.static.data(name="x_data", shape=[2, 1], dtype=t)
                    exe = paddle.static.Executor(place)
                    output = paddle.nn.BatchNorm1D(1)(x_data)
                    exe.run(startup_program)
                    res = exe.run(main_program, feed=feed, fetch_list=[output])
                    expect = np.array([[0.99988127], [-0.99988127]])
                    compare(res[0], expect)


@pytest.mark.api_nn_BatchNorm1D_parameters
def test_static2():
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

            main_program = paddle.static.Program()
            startup_program = paddle.static.Program()
            main_program.random_seed = seed
            startup_program.random_seed = seed
            x_data = np.array([[[0.6964692, 0.28613934, 0.22685145]], [[0.5513148, 0.71946895, 0.42310646]]]).astype(t)
            feed = {"x_data": x_data}
            with paddle.utils.unique_name.guard():
                with paddle.static.program_guard(main_program=main_program, startup_program=startup_program):
                    x_data = paddle.static.data(name="x_data", shape=[2, 1, 3], dtype=t)
                    exe = paddle.static.Executor(place)
                    output = paddle.nn.BatchNorm1D(1)(x_data)
                    exe.run(startup_program)
                    res = exe.run(main_program, feed=feed, fetch_list=[output])
                    expect = np.array([[[1.1251787, -1.0467086, -1.3605211]], [[0.35687235, 1.2469171, -0.32173783]]])
                    compare(res[0], expect)


@pytest.mark.api_nn_BatchNorm1D_parameters
def test_static3():
    """
    input=2-D Tensor, momentum=1
    :return:
    """
    for place in places:
        for t in types:
            if t == np.float64:
                paddle.set_default_dtype("float64")
            elif t == np.float32:
                paddle.set_default_dtype("float32")

            main_program = paddle.static.Program()
            startup_program = paddle.static.Program()
            main_program.random_seed = seed
            startup_program.random_seed = seed
            x_data = np.array([[0.6964692], [0.28613934]]).astype(t)
            feed = {"x_data": x_data}
            with paddle.utils.unique_name.guard():
                with paddle.static.program_guard(main_program=main_program, startup_program=startup_program):
                    x_data = paddle.static.data(name="x_data", shape=[2, 1], dtype=t)
                    exe = paddle.static.Executor(place)
                    output = paddle.nn.BatchNorm1D(num_features=1, momentum=1)(x_data)
                    exe.run(startup_program)
                    res = exe.run(main_program, feed=feed, fetch_list=[output])
                    expect = np.array([[0.99988127], [-0.99988127]])
                    compare(res[0], expect)


@pytest.mark.api_nn_BatchNorm1D_parameters
def test_static4():
    """
    input=3-D Tensor, momentum=0
    :return:
    """
    for place in places:
        for t in types:
            if t == np.float64:
                paddle.set_default_dtype("float64")
            elif t == np.float32:
                paddle.set_default_dtype("float32")

            main_program = paddle.static.Program()
            startup_program = paddle.static.Program()
            main_program.random_seed = seed
            startup_program.random_seed = seed
            x_data = np.array([[[0.6964692, 0.28613934, 0.22685145]], [[0.5513148, 0.71946895, 0.42310646]]]).astype(t)
            feed = {"x_data": x_data}
            with paddle.utils.unique_name.guard():
                with paddle.static.program_guard(main_program=main_program, startup_program=startup_program):
                    x_data = paddle.static.data(name="x_data", shape=[2, 1, 3], dtype=t)
                    exe = paddle.static.Executor(place)
                    output = paddle.nn.BatchNorm1D(num_features=1, momentum=0)(x_data)
                    exe.run(startup_program)
                    res = exe.run(main_program, feed=feed, fetch_list=[output])
                    expect = np.array([[[1.1251787, -1.0467086, -1.3605211]], [[0.35687235, 1.2469171, -0.32173783]]])
                    compare(res[0], expect)


@pytest.mark.api_nn_BatchNorm1D_parameters
def test_static5():
    """
    input=2-D Tensor, momentum=1, epsilon<=1e-3
    :return:
    """
    for place in places:
        for t in types:
            if t == np.float64:
                paddle.set_default_dtype("float64")
            elif t == np.float32:
                paddle.set_default_dtype("float32")

            main_program = paddle.static.Program()
            startup_program = paddle.static.Program()
            main_program.random_seed = seed
            startup_program.random_seed = seed
            x_data = np.array([[0.6964692], [0.28613934]]).astype(t)
            feed = {"x_data": x_data}
            with paddle.utils.unique_name.guard():
                with paddle.static.program_guard(main_program=main_program, startup_program=startup_program):
                    x_data = paddle.static.data(name="x_data", shape=[2, 1], dtype=t)
                    exe = paddle.static.Executor(place)
                    output = paddle.nn.BatchNorm1D(num_features=1, momentum=1, epsilon=1e-5)(x_data)
                    exe.run(startup_program)
                    res = exe.run(main_program, feed=feed, fetch_list=[output])
                    expect = np.array([[0.99988127], [-0.99988127]])
                    compare(res[0], expect)


@pytest.mark.api_nn_BatchNorm1D_parameters
def test_static6():
    """
    input=3-D Tensor, momentum=0, epsilon=<1e-3
    :return:
    """
    for place in places:
        for t in types:
            if t == np.float64:
                paddle.set_default_dtype("float64")
            elif t == np.float32:
                paddle.set_default_dtype("float32")

            main_program = paddle.static.Program()
            startup_program = paddle.static.Program()
            main_program.random_seed = seed
            startup_program.random_seed = seed
            x_data = np.array([[[0.6964692, 0.28613934, 0.22685145]], [[0.5513148, 0.71946895, 0.42310646]]]).astype(t)
            feed = {"x_data": x_data}
            with paddle.utils.unique_name.guard():
                with paddle.static.program_guard(main_program=main_program, startup_program=startup_program):
                    x_data = paddle.static.data(name="x_data", shape=[2, 1, 3], dtype=t)
                    exe = paddle.static.Executor(place)
                    output = paddle.nn.BatchNorm1D(num_features=1, momentum=0, epsilon=1e-5)(x_data)
                    exe.run(startup_program)
                    res = exe.run(main_program, feed=feed, fetch_list=[output])
                    expect = np.array([[[1.1251787, -1.0467086, -1.3605211]], [[0.35687235, 1.2469171, -0.32173783]]])
                    compare(res[0], expect)


@pytest.mark.api_nn_BatchNorm1D_parameters
def test_static7():
    """
    input=3-D Tensor, momentum=0, epsilon=<1e-3, weight_attr=attrs, bias_attr=attrs,
    :return: res
    """
    for place in places:
        for t in types:
            if t == np.float64:
                paddle.set_default_dtype("float64")
            elif t == np.float32:
                paddle.set_default_dtype("float32")

            w_param_attrs = paddle.ParamAttr(
                name="w1_weight", learning_rate=0.5, regularizer=paddle.regularizer.L2Decay(1.0), trainable=True
            )
            b_param_attrs = paddle.ParamAttr(
                name="b1_weight", learning_rate=0.5, regularizer=paddle.regularizer.L2Decay(1.0), trainable=True
            )
            main_program = paddle.static.Program()
            startup_program = paddle.static.Program()
            main_program.random_seed = seed
            startup_program.random_seed = seed
            x_data = np.array([[[0.6964692, 0.28613934, 0.22685145]], [[0.5513148, 0.71946895, 0.42310646]]]).astype(t)
            feed = {"x_data": x_data}
            with paddle.utils.unique_name.guard():
                with paddle.static.program_guard(main_program=main_program, startup_program=startup_program):
                    x_data = paddle.static.data(name="x_data", shape=[2, 1, 3], dtype=t)
                    exe = paddle.static.Executor(place)
                    output = paddle.nn.BatchNorm1D(
                        num_features=1, momentum=0, epsilon=1e-5, weight_attr=w_param_attrs, bias_attr=b_param_attrs
                    )(x_data)
                    exe.run(startup_program)
                    res = exe.run(main_program, feed=feed, fetch_list=[output])
                    expect = np.array([[[1.1251787, -1.0467086, -1.3605211]], [[0.35687235, 1.2469171, -0.32173783]]])
                    compare(res[0], expect)


@pytest.mark.api_nn_BatchNorm1D_parameters
def test_static8():
    """
    input=3-D Tensor, momentum=0, epsilon=<1e-3, weight_attr=attrs, bias_attr=attrs, data_format='NCL'
    :return: res
    """
    for place in places:
        for t in types:
            if t == np.float64:
                paddle.set_default_dtype("float64")
            elif t == np.float32:
                paddle.set_default_dtype("float32")

            w_param_attrs = paddle.ParamAttr(
                name="w1_weight", learning_rate=0.5, regularizer=paddle.regularizer.L2Decay(1.0), trainable=True
            )
            b_param_attrs = paddle.ParamAttr(
                name="b1_weight", learning_rate=0.5, regularizer=paddle.regularizer.L2Decay(1.0), trainable=True
            )
            main_program = paddle.static.Program()
            startup_program = paddle.static.Program()
            main_program.random_seed = seed
            startup_program.random_seed = seed
            x_data = np.array([[[0.6964692, 0.28613934, 0.22685145]], [[0.5513148, 0.71946895, 0.42310646]]]).astype(t)
            feed = {"x_data": x_data}
            with paddle.utils.unique_name.guard():
                with paddle.static.program_guard(main_program=main_program, startup_program=startup_program):
                    x_data = paddle.static.data(name="x_data", shape=[2, 1, 3], dtype=t)
                    exe = paddle.static.Executor(place)
                    output = paddle.nn.BatchNorm1D(
                        num_features=1,
                        momentum=0,
                        epsilon=1e-5,
                        weight_attr=w_param_attrs,
                        bias_attr=b_param_attrs,
                        data_format="NCL",
                    )(x_data)
                    exe.run(startup_program)
                    res = exe.run(main_program, feed=feed, fetch_list=[output])
                    expect = np.array([[[1.1251787, -1.0467086, -1.3605211]], [[0.35687235, 1.2469171, -0.32173783]]])
                    compare(res[0], expect)
