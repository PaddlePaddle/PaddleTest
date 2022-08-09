#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_batchnorm
"""
import copy
from apibase import APIBase
from apibase import randtool
import paddle
import pytest
import numpy as np


class TestBatchNorm(APIBase):
    """
    test
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32]
        self.delta = 1e-5
        # self.rtol = 1e-3
        # self.debug = True
        # self.static = True
        # enable check grad
        # self.enable_backward = False


obj = TestBatchNorm(paddle.nn.BatchNorm)


def cal_batchnorm(x, num_channels, epsilon=1e-5, weight=None, bias=None, data_layout="NCHW"):
    """
    calculate batchnorm
    """
    if data_layout == "NCHW":
        data = copy.deepcopy(x)
    elif data_layout == "NHWC":
        data = copy.deepcopy(np.transpose(x, (0, 3, 1, 2)))
        # print(data.shape)

    for i in range(num_channels):
        a = data[:, i]
        a_mean = a.mean()
        a_var = a.var()
        data[:, i] = (a - a_mean) / ((a_var + epsilon) ** 0.5)
        if weight is not None:
            data[:, i] = weight[i] * data[:, i]
        if bias is not None:
            data[:, i] = data[:, i] + bias[i]

    if data_layout == "NCHW":
        return data
    elif data_layout == "NHWC":
        return np.transpose(data, (0, 2, 3, 1))


@pytest.mark.api_nn_BatchNorm_vartype
def test_batchnorm_base():
    """
    base
    x: 2-D tensor
    """
    x = randtool("float", -2, 10, (2, 3))
    res = cal_batchnorm(x, num_channels=3)
    obj.base(res=res, num_channels=3, data=x)


@pytest.mark.api_nn_BatchNorm_parameters
def test_batchnorm0():
    """
    default
    x: 3-D tensor
    """
    x = randtool("float", -4, 4, (3, 4, 5))
    res = cal_batchnorm(x, num_channels=4)
    obj.run(res=res, num_channels=4, data=x)


@pytest.mark.api_nn_BatchNorm_parameters
def test_batchnorm1():
    """
    default
    x: 4-D tensor
    """
    x = randtool("float", -4, 4, (3, 4, 5, 6))
    res = cal_batchnorm(x, num_channels=4)
    obj.run(res=res, num_channels=4, data=x)


@pytest.mark.api_nn_BatchNorm_parameters
def test_batchnorm2():
    """
    default
    x: 5-D tensor
    """
    x = randtool("float", -4, 4, (3, 4, 5, 6, 7))
    res = cal_batchnorm(x, num_channels=4)
    obj.run(res=res, num_channels=4, data=x)


@pytest.mark.api_nn_BatchNorm_parameters
def test_batchnorm3():
    """
    x: 4-D tensor
    data_layout='NHWC'
    """
    x = randtool("float", -4, 4, (3, 4, 5, 6))
    res = cal_batchnorm(x, num_channels=6, data_layout="NHWC")
    obj.run(res=res, num_channels=6, data=x, data_layout="NHWC")


@pytest.mark.api_nn_BatchNorm_parameters
def test_batchnorm4():
    """
    x: 4-D tensor
    epsilon = 1e-4
    """
    x = randtool("float", -4, 4, (3, 4, 5, 6))
    res = cal_batchnorm(x, num_channels=4, epsilon=1e-4)
    obj.run(res=res, num_channels=4, data=x, epsilon=1e-4)


@pytest.mark.api_nn_BatchNorm_parameters
def test_batchnorm5():
    """
    x: 4-D tensor
    weight: constant(4)
    """
    x = randtool("float", -4, 4, (3, 4, 5, 6))
    weight = np.array([4] * 4)
    res = cal_batchnorm(x, num_channels=4, weight=weight)
    obj.run(res=res, num_channels=4, data=x, param_attr=paddle.nn.initializer.Constant(4))


@pytest.mark.api_nn_BatchNorm_parameters
def test_batchnorm6():
    """
    x: 4-D tensor
    weight: constant(4)
    bias: constant(2)
    """
    x = randtool("float", -4, 4, (3, 4, 5, 6))
    weight, bias = np.array([4] * 4), np.array([2] * 4)
    res = cal_batchnorm(x, num_channels=4, weight=weight, bias=bias)
    obj.run(
        res=res,
        num_channels=4,
        data=x,
        param_attr=paddle.nn.initializer.Constant(4),
        bias_attr=paddle.nn.initializer.Constant(2),
    )


@pytest.mark.api_nn_BatchNorm_parameters
def test_batchnorm7():
    """
    x: 4-D tensor
    act: relu
    """
    x = randtool("float", -4, 4, (3, 8, 2, 4))
    res = cal_batchnorm(x, num_channels=8)
    res = np.where(res > 0, res, 0)
    obj.run(res=res, num_channels=8, data=x, act="relu")


@pytest.mark.api_nn_BatchNorm_parameters
def test_batchnorm8():
    """
    x: 4-D tensor
    act: sigmoid
    """
    x = randtool("float", -4, 4, (3, 8, 2, 4))
    res = cal_batchnorm(x, num_channels=8)
    res = 1 / (1 + np.exp(-res))
    obj.run(res=res, num_channels=8, data=x, act="sigmoid")


@pytest.mark.api_nn_BatchNorm_parameters
def test_batchnorm9():
    """
    x: 4-D tensor
    use_global_stats = True: -->> 0, 1
    """
    x = randtool("float", -4, 4, (3, 2, 4, 6))
    res = x / ((1 + 1e-5) ** 0.5)
    obj.run(res=res, num_channels=2, data=x, param_attr=paddle.ParamAttr(learning_rate=0.1), use_global_stats=True)


@pytest.mark.api_nn_BatchNorm_parameters
def test_batchnorm10():
    """
    x: 4-D tensor
    is_test = True
    """
    if paddle.is_compiled_with_cuda():
        places = [paddle.CPUPlace(), paddle.CUDAPlace(0)]
    else:
        places = [paddle.CPUPlace()]

    for place in places:
        x = randtool("float", -4, 4, (2, 3, 4, 5)).astype("float32")
        res = x / ((1 + 1e-5) ** 0.5)

        paddle.disable_static(place)
        batch_norm = paddle.nn.BatchNorm(3, is_test=True)
        batch_norm.eval()
        api_dynamic_res = batch_norm(paddle.to_tensor(x))

        paddle.enable_static()
        main_program = paddle.static.default_main_program()
        startup_program = paddle.static.default_startup_program()
        with paddle.static.program_guard(main_program=main_program, startup_program=startup_program):
            x_data = paddle.static.data(name="x_data", shape=[2, 3, 4, 5], dtype="float32")
            exe = paddle.static.Executor(place)
            batch_norm = paddle.nn.BatchNorm(num_channels=3, is_test=True)
            batch_norm.eval()
            output = batch_norm(x_data)
            exe.run(startup_program)
            api_static_res = exe.run(main_program, feed={"x_data": x}, fetch_list=[output])

            assert np.allclose(res, api_dynamic_res.numpy())
            assert np.allclose(api_dynamic_res.numpy(), api_static_res[0])


@pytest.mark.api_nn_BatchNorm_parameters
def test_batchnorm11():
    """
    x: 4-D tensor
    trainable_statistics = True
    """
    if paddle.is_compiled_with_cuda():
        places = [paddle.CPUPlace(), paddle.CUDAPlace(0)]
    else:
        places = [paddle.CPUPlace()]

    for place in places:
        x = randtool("float", -4, 4, (2, 3, 4, 5)).astype("float32")
        res = cal_batchnorm(x, 3)

        paddle.disable_static(place)
        batch_norm = paddle.nn.BatchNorm(3, trainable_statistics=True)
        batch_norm.eval()
        api_dynamic_res = batch_norm(paddle.to_tensor(x))

        paddle.enable_static()
        main_program = paddle.static.default_main_program()
        startup_program = paddle.static.default_startup_program()
        with paddle.static.program_guard(main_program=main_program, startup_program=startup_program):
            x_data = paddle.static.data(name="x_data", shape=[2, 3, 4, 5], dtype="float32")
            exe = paddle.static.Executor(place)
            batch_norm = paddle.nn.BatchNorm(num_channels=3, trainable_statistics=True)
            batch_norm.eval()
            output = batch_norm(x_data)
            exe.run(startup_program)
            api_static_res = exe.run(main_program, feed={"x_data": x}, fetch_list=[output])

            assert np.allclose(res, api_dynamic_res.numpy())
            assert np.allclose(api_dynamic_res.numpy(), api_static_res[0])


@pytest.mark.api_nn_BatchNorm_parameters
def test_batchnorm12():
    """
    dtype = float64
    x: 4-D tensor
    """
    obj.types = [np.float64]
    x = randtool("float", -4, 4, (3, 4, 5, 6))
    res = cal_batchnorm(x, num_channels=4)
    obj.run(res=res, num_channels=4, dtype="float64", data=x)


@pytest.mark.api_nn_BatchNorm_parameters
def test_batchnorm13():
    """
    paddle.set_default_dtype("float16")
    after batchnorm api, paddle.get_default_dtype() should be float16
    """
    paddle.disable_static()
    paddle.set_default_dtype("float16")
    np.random.seed(123)
    x_data = np.random.random(size=(2, 1, 2, 3)).astype("float32")
    x = paddle.to_tensor(x_data)
    batch_norm = paddle.nn.BatchNorm2D(1)
    batch_norm_out = batch_norm(x)
    out = paddle.get_default_dtype()
    assert batch_norm_out.dtype == paddle.float32
    assert out == "float16"
