#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_sequence_mask
"""
import pytest
import paddle
import numpy as np
from apibase import compare


@pytest.mark.api_nn_sequence_mask_parameters
def test_sequence_mask1():
    """
    inputs = paddle.to_tensor([10, 9, 8], dtype='int64')
    out = paddle.nn.functional.sequence_mask(inputs)
    """
    x_tensor = paddle.to_tensor([10, 9, 8], dtype="int64")
    res = paddle.nn.functional.sequence_mask(x_tensor)
    exp = np.array(
        [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1, 1, 1, 0, 0]]
    ).astype(np.int64)
    compare(np.array(res), exp)


@pytest.mark.api_nn_sequence_mask_parameters
def test_sequence_mask2():
    """
    inputs = paddle.to_tensor([10, 20, 8, 5, 5, 6, 6, 8], dtype='int32')
    out = paddle.nn.functional.sequence_mask(inputs)
    """
    x_tensor = paddle.to_tensor([10, 20, 8, 5, 5, 6, 6, 8], dtype="int32")
    res = paddle.nn.functional.sequence_mask(x_tensor)
    exp = np.array(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    ).astype(np.int32)
    compare(np.array(res), exp)


@pytest.mark.api_nn_sequence_mask_parameters
def test_sequence_mask3():
    """
    inputs = paddle.to_tensor([10., 9., 8.], dtype='float64')
    out = paddle.nn.functional.sequence_mask(inputs)
    """
    x_tensor = paddle.to_tensor([10.0, 9.0, 8.0], dtype="float64")
    res = paddle.nn.functional.sequence_mask(x_tensor, maxlen=20)
    exp = np.array(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    ).astype(np.float64)
    compare(np.array(res), exp)


@pytest.mark.api_nn_sequence_mask_parameters
def test_sequence_mask4():
    """
    inputs = paddle.to_tensor([10., 9., 8.], dtype='float32')
    out = paddle.nn.functional.sequence_mask(inputs)
    """
    x_tensor = paddle.to_tensor([10.0, 9.0, 8.0])
    res = paddle.nn.functional.sequence_mask(x_tensor, dtype=np.float64)
    exp = np.array(
        [
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
        ]
    ).astype(np.float64)
    compare(np.array(res), exp)


@pytest.mark.api_nn_sequence_mask_parameters
def test_sequence_mask5():
    """
    inputs = paddle.to_tensor([10, 9, 8], dtype='int32')
    out = paddle.nn.functional.sequence_mask(inputs)
    """
    x_tensor = paddle.to_tensor([10, 9, 8])
    res = paddle.nn.functional.sequence_mask(x_tensor, dtype=np.float64)
    exp = np.array(
        [
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
        ]
    ).astype(np.float64)
    compare(np.array(res), exp)


@pytest.mark.api_nn_sequence_mask_parameters
def test_sequence_mask6():
    """
    inputs = paddle.to_tensor([[3, 2, 1],
                               [6, 5, 4]], dtype='int32')
    out = paddle.nn.functional.sequence_mask(inputs, maxlen=30, dtype=np.int32)
    """
    x_tensor = paddle.to_tensor([[3, 2, 1], [6, 5, 4]])
    res = paddle.nn.functional.sequence_mask(x_tensor, maxlen=30, dtype=np.int32)
    exp = np.array(
        [
            [
                [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            [
                [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
        ]
    ).astype(np.int32)
    compare(np.array(res), exp)


@pytest.mark.api_nn_sequence_mask_parameters
def test_sequence_mask7():
    """
    inputs = paddle.to_tensor(np.ones((2, 2, 3, 3, 3, 3, 1, 1)) * 2, dtype='int32')
    out = paddle.nn.functional.sequence_mask(inputs, maxlen=5, dtype=np.int32)
    """
    x_tensor = paddle.to_tensor(np.ones((2, 2, 3, 3, 3)) * 2)
    res = paddle.nn.functional.sequence_mask(x_tensor, maxlen=5, dtype=np.int32)
    exp = np.array(
        [
            [
                [
                    [
                        [[1, 1, 0, 0, 0], [1, 1, 0, 0, 0], [1, 1, 0, 0, 0]],
                        [[1, 1, 0, 0, 0], [1, 1, 0, 0, 0], [1, 1, 0, 0, 0]],
                        [[1, 1, 0, 0, 0], [1, 1, 0, 0, 0], [1, 1, 0, 0, 0]],
                    ],
                    [
                        [[1, 1, 0, 0, 0], [1, 1, 0, 0, 0], [1, 1, 0, 0, 0]],
                        [[1, 1, 0, 0, 0], [1, 1, 0, 0, 0], [1, 1, 0, 0, 0]],
                        [[1, 1, 0, 0, 0], [1, 1, 0, 0, 0], [1, 1, 0, 0, 0]],
                    ],
                    [
                        [[1, 1, 0, 0, 0], [1, 1, 0, 0, 0], [1, 1, 0, 0, 0]],
                        [[1, 1, 0, 0, 0], [1, 1, 0, 0, 0], [1, 1, 0, 0, 0]],
                        [[1, 1, 0, 0, 0], [1, 1, 0, 0, 0], [1, 1, 0, 0, 0]],
                    ],
                ],
                [
                    [
                        [[1, 1, 0, 0, 0], [1, 1, 0, 0, 0], [1, 1, 0, 0, 0]],
                        [[1, 1, 0, 0, 0], [1, 1, 0, 0, 0], [1, 1, 0, 0, 0]],
                        [[1, 1, 0, 0, 0], [1, 1, 0, 0, 0], [1, 1, 0, 0, 0]],
                    ],
                    [
                        [[1, 1, 0, 0, 0], [1, 1, 0, 0, 0], [1, 1, 0, 0, 0]],
                        [[1, 1, 0, 0, 0], [1, 1, 0, 0, 0], [1, 1, 0, 0, 0]],
                        [[1, 1, 0, 0, 0], [1, 1, 0, 0, 0], [1, 1, 0, 0, 0]],
                    ],
                    [
                        [[1, 1, 0, 0, 0], [1, 1, 0, 0, 0], [1, 1, 0, 0, 0]],
                        [[1, 1, 0, 0, 0], [1, 1, 0, 0, 0], [1, 1, 0, 0, 0]],
                        [[1, 1, 0, 0, 0], [1, 1, 0, 0, 0], [1, 1, 0, 0, 0]],
                    ],
                ],
            ],
            [
                [
                    [
                        [[1, 1, 0, 0, 0], [1, 1, 0, 0, 0], [1, 1, 0, 0, 0]],
                        [[1, 1, 0, 0, 0], [1, 1, 0, 0, 0], [1, 1, 0, 0, 0]],
                        [[1, 1, 0, 0, 0], [1, 1, 0, 0, 0], [1, 1, 0, 0, 0]],
                    ],
                    [
                        [[1, 1, 0, 0, 0], [1, 1, 0, 0, 0], [1, 1, 0, 0, 0]],
                        [[1, 1, 0, 0, 0], [1, 1, 0, 0, 0], [1, 1, 0, 0, 0]],
                        [[1, 1, 0, 0, 0], [1, 1, 0, 0, 0], [1, 1, 0, 0, 0]],
                    ],
                    [
                        [[1, 1, 0, 0, 0], [1, 1, 0, 0, 0], [1, 1, 0, 0, 0]],
                        [[1, 1, 0, 0, 0], [1, 1, 0, 0, 0], [1, 1, 0, 0, 0]],
                        [[1, 1, 0, 0, 0], [1, 1, 0, 0, 0], [1, 1, 0, 0, 0]],
                    ],
                ],
                [
                    [
                        [[1, 1, 0, 0, 0], [1, 1, 0, 0, 0], [1, 1, 0, 0, 0]],
                        [[1, 1, 0, 0, 0], [1, 1, 0, 0, 0], [1, 1, 0, 0, 0]],
                        [[1, 1, 0, 0, 0], [1, 1, 0, 0, 0], [1, 1, 0, 0, 0]],
                    ],
                    [
                        [[1, 1, 0, 0, 0], [1, 1, 0, 0, 0], [1, 1, 0, 0, 0]],
                        [[1, 1, 0, 0, 0], [1, 1, 0, 0, 0], [1, 1, 0, 0, 0]],
                        [[1, 1, 0, 0, 0], [1, 1, 0, 0, 0], [1, 1, 0, 0, 0]],
                    ],
                    [
                        [[1, 1, 0, 0, 0], [1, 1, 0, 0, 0], [1, 1, 0, 0, 0]],
                        [[1, 1, 0, 0, 0], [1, 1, 0, 0, 0], [1, 1, 0, 0, 0]],
                        [[1, 1, 0, 0, 0], [1, 1, 0, 0, 0], [1, 1, 0, 0, 0]],
                    ],
                ],
            ],
        ]
    ).astype(np.int32)
    compare(np.array(res), exp)


@pytest.mark.api_nn_sequence_mask_parameters
def test_sequence_mask8():
    """
    paddle.enable_static()
    inputs = np.array([5., 5., 5.]).astype(np.float32)
    out = paddle.nn.functional.sequence_mask(inputs, dtype=np.float64, maxlen=11, name='m')
    """
    paddle.enable_static()
    image = paddle.static.data(name="x8", shape=[3, 1], dtype="float32")
    x_tensor = np.array([5.0, 3.0, 1.0]).astype(np.float32)
    mask = paddle.nn.functional.sequence_mask(image, dtype=np.float64, maxlen=11, name="m")
    exe = paddle.static.Executor()
    prog = paddle.static.default_main_program()
    exe.run(paddle.static.default_startup_program())

    res = exe.run(prog, feed={"x8": x_tensor}, fetch_list=[mask])
    exp = np.array(
        [
            [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    ).astype(np.float64)
    compare(res[0], exp)


@pytest.mark.api_nn_sequence_mask_parameters
def sequence_mask(x_len, max_len=None, dtype="float32"):
    """
    create a def to test broadcast
    """
    max_len = max_len or x_len.max()
    x_len = paddle.unsqueeze(x_len, -1)
    row_vector = paddle.arange(max_len)
    mask = row_vector < x_len
    mask = paddle.cast(mask, dtype)
    return mask


@pytest.mark.api_nn_sequence_mask_parameters
def test_sequence_mask9():
    """
    diy def sequence_mask, test broadcast
    """
    paddle.disable_static()
    res = sequence_mask(paddle.to_tensor([3, 4]))
    exp = np.array([[1, 1, 1, 0], [1, 1, 1, 1]]).astype(np.float32)
    compare(np.array(res), exp)
