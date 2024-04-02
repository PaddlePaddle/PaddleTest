#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_maxunpool2d
"""

import paddle
import pytest
import numpy as np


def cal_max_unpool(x_data, indices, kernel_size, stride=None, padding=0, output_size=None):
    """
    calculate max_unpool2d
    """

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    if not stride:
        stride = kernel_size

    if isinstance(padding, int):
        padding = (padding, padding)

    if output_size is None:
        n, c, h, w = x_data.shape
        out_h = (h - 1) * stride[0] - 2 * padding[0] + kernel_size[0]
        out_w = (w - 1) * stride[1] - 2 * padding[1] + kernel_size[1]
        output_size = (n, c, out_h, out_w)
    else:
        if len(output_size) == len(kernel_size) + 2:
            output_size = output_size[2:]

    out = np.zeros(output_size)

    for i in range(indices.shape[0]):
        for j in range(indices.shape[1]):
            for k in range(indices.shape[2]):
                for m in range(indices.shape[3]):
                    indices[i, j, k, m] = (
                        (out.shape[1] * out.shape[2] * out.shape[3]) * i
                        + (out.shape[2] * out.shape[3]) * j
                        + indices[i, j, k, m]
                    )

    flatten_out = out.flatten()
    flatten_indices = indices.flatten()
    flatten_x = x_data.flatten()
    for i in range(flatten_indices.shape[0]):
        flatten_out[int(flatten_indices[i])] = flatten_x[i]

    out = np.reshape(flatten_out, out.shape)
    return out


def cal_static_result(
    place, dtype, x, indices, kernel_size, stride=None, padding=0, data_format="NCHW", output_size=None
):
    """
    calculate static forward
    """
    paddle.enable_static()
    main_program, strartup_program = paddle.static.Program(), paddle.static.Program()
    with paddle.static.program_guard(main_program=main_program, startup_program=strartup_program):
        data1 = paddle.static.data(name="data1", shape=x.shape, dtype=dtype)
        data2 = paddle.static.data(name="data2", shape=indices.shape, dtype="int32")
        max_unpool = paddle.nn.MaxUnPool2D(kernel_size, stride, padding, data_format, output_size)
        output = max_unpool(data1, data2)
        exe = paddle.static.Executor(place=place)
        exe.run(strartup_program)
        res = exe.run(main_program, feed={"data1": x, "data2": indices}, fetch_list=[output])
    return res[0]


if paddle.is_compiled_with_cuda():
    places = [paddle.CPUPlace(), paddle.CUDAPlace(0)]
else:
    places = [paddle.CPUPlace()]


@pytest.mark.api_nn_max_unpool2d_parameters
def test_local_response_norm0():
    """
    base
    kernel_size = 2
    """
    types = ["float64", "float32"]
    for place in places:
        for dtype in types:
            paddle.disable_static(place=place)
            np.random.seed(22)
            x_data = np.random.rand(2, 4, 40, 40)
            x = paddle.to_tensor(x_data)
            pool = paddle.nn.MaxPool2D(kernel_size=2, return_mask=True)
            y, indices = pool(x)
            y = y.astype(dtype)

            r_unpool = paddle.nn.MaxUnPool2D(2)
            dynamic_res = r_unpool(y, indices)
            y, indices = y.numpy(), indices.numpy()
            static_res = cal_static_result(place, dtype, y, indices, 2)

            expect = cal_max_unpool(y, indices, 2)

            assert np.allclose(static_res, dynamic_res.numpy())
            assert np.allclose(static_res, expect)


@pytest.mark.api_nn_max_unpool2d_parameters
def test_local_response_norm1():
    """
    kernel_size = (2, 4)
    """
    types = ["float64", "float32"]
    for place in places:
        for dtype in types:
            paddle.disable_static(place=place)
            np.random.seed(22)
            x_data = np.random.rand(2, 4, 40, 40)
            x = paddle.to_tensor(x_data)
            pool = paddle.nn.MaxPool2D(kernel_size=(2, 4), return_mask=True)
            y, indices = pool(x)
            y = y.astype(dtype)

            r_unpool = paddle.nn.MaxUnPool2D((2, 4))
            dynamic_res = r_unpool(y, indices)
            y, indices = y.numpy(), indices.numpy()
            static_res = cal_static_result(place, dtype, y, indices, (2, 4))

            expect = cal_max_unpool(y, indices, (2, 4))

            assert np.allclose(static_res, dynamic_res.numpy())
            assert np.allclose(static_res, expect)


@pytest.mark.api_nn_max_unpool2d_parameters
def test_local_response_norm2():
    """
    kernel_size = 4
    padding = 2
    """
    types = ["float64", "float32"]
    for place in places:
        for dtype in types:
            paddle.disable_static(place=place)
            np.random.seed(22)
            x_data = np.random.rand(2, 4, 40, 40)
            x = paddle.to_tensor(x_data)
            pool = paddle.nn.MaxPool2D(kernel_size=4, padding=2, return_mask=True)
            y, indices = pool(x)
            y = y.astype(dtype)

            r_unpool = paddle.nn.MaxUnPool2D(4, padding=2)
            dynamic_res = r_unpool(y, indices)
            y, indices = y.numpy(), indices.numpy()
            static_res = cal_static_result(place, dtype, y, indices, 4, padding=2)

            expect = cal_max_unpool(y, indices, 4, padding=2)

            assert np.allclose(static_res, dynamic_res.numpy())
            assert np.allclose(static_res, expect)


@pytest.mark.api_nn_max_unpool2d_parameters
def test_local_response_norm3():
    """
    kernel_size = 4
    padding = 2
    stride = 2
    """
    types = ["float64", "float32"]
    for place in places:
        for dtype in types:
            paddle.disable_static(place=place)
            np.random.seed(22)
            x_data = np.random.rand(2, 4, 40, 40)
            x = paddle.to_tensor(x_data)
            pool = paddle.nn.MaxPool2D(kernel_size=4, padding=2, stride=2, return_mask=True)
            y, indices = pool(x)
            y = y.astype(dtype)

            r_unpool = paddle.nn.MaxUnPool2D(4, padding=2, stride=2)
            dynamic_res = r_unpool(y, indices)
            y, indices = y.numpy(), indices.numpy()
            static_res = cal_static_result(place, dtype, y, indices, 4, padding=2, stride=2)

            expect = cal_max_unpool(y, indices, 4, padding=2, stride=(2, 2))

            assert np.allclose(static_res, dynamic_res.numpy())
            assert np.allclose(static_res, expect)


# @pytest.mark.api_nn_max_unpool2d_parameters
# def test_local_response_norm4():
#     """
#     kernel_size = 4
#     padding = 2
#     stride = 2
#     data_format = NHWC
#     """
#     types = ['float64', 'float32']
#     for place in places:
#         for dtype in types:
#             paddle.disable_static(place=place)
#             np.random.seed(22)
#             x_data = np.random.rand(2, 40, 40, 4)
#             x = paddle.to_tensor(x_data)
#             y = np.random.randint(0, 1600, (2, 40, 40, 4)).astype('int32')
#             indices = paddle.to_tensor(y, dtype='int32')
#             # pool = paddle.nn.MaxPool2D(kernel_size=4, padding=2, stride=2, data_format='NHWC', return_mask=True)
#             # y, indices = pool(x)
#             # y = y.astype(dtype)
#
#             r_unpool = paddle.nn.MaxUnPool2D(4, padding=2, stride=2, data_format='NHWC')
#             dynamic_res = r_unpool(x, indices)
#             # y, indices = y.numpy(), indices.numpy()
#             static_res = cal_static_result(place, dtype, x_data, y, 4, padding=2, stride=2, data_format='NHWC')
#
#             # expect = cal_max_unpool(y, indices, 4, padding=2, stride=(2, 2))
#
#             assert np.allclose(static_res, dynamic_res.numpy())
#             # assert np.allclose(static_res, expect)


@pytest.mark.api_nn_max_unpool2d_parameters
def test_local_response_norm5():
    """
    base
    kernel_size = 2
    set output_size
    """
    types = ["float64", "float32"]
    for place in places:
        for dtype in types:
            paddle.disable_static(place=place)
            np.random.seed(22)
            x_data = np.random.rand(2, 4, 40, 40)
            x = paddle.to_tensor(x_data)
            pool = paddle.nn.MaxPool2D(kernel_size=2, return_mask=True)
            y, indices = pool(x)
            y = y.astype(dtype)

            r_unpool = paddle.nn.MaxUnPool2D(2, output_size=(2, 4, 40, 40))
            dynamic_res = r_unpool(y, indices)
            y, indices = y.numpy(), indices.numpy()
            static_res = cal_static_result(place, dtype, y, indices, 2, output_size=(2, 4, 40, 40))

            expect = cal_max_unpool(y, indices, 2)

            assert np.allclose(static_res, dynamic_res.numpy())
            assert np.allclose(static_res, expect)
