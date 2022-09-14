#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_functional_max_pool2d
"""
from apibase import APIBase
from apibase import randtool
import paddle
import pytest
import numpy as np


# class TestFunctionalMaxUnPool2d(APIBase):
#     """
#     test
#     """
#
#     def hook(self):
#         """
#         implement
#         """
#         self.types = [np.float32, np.float64]
#         self.delta = 1e-3
#         # self.debug = True
#         # self.static = True
#         # enable check grad
#         self.enable_backward = False
#
#
# obj = TestFunctionalMaxUnPool2d(paddle.nn.functional.max_unpool2d)


@pytest.mark.api_nn_max_pool2d_vartype
def test_max_unpool2d_base():
    """
    base
    """
    np.random.seed(33)
    paddle.seed(33)
    x = randtool("float", -10, 10, [1, 1, 3, 3])
    indices = randtool("int", 0, 10, [1, 1, 3, 3]).astype("int32")
    kernel_size = 2
    padding = 0
    res = np.array(
        [
            [
                [
                    [-1.78118394, 0.0, 0.0, 0.0, 9.06504063, 0.0],
                    [-1.00049158, -5.02979745, 3.60901609, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            ]
        ]
    )
    exp = paddle.nn.functional.max_unpool2d(
        x=paddle.to_tensor(x), indices=paddle.to_tensor(indices), kernel_size=kernel_size, padding=padding
    )
    assert np.allclose(exp.numpy(), res)


@pytest.mark.api_nn_max_pool2d_parameters
def test_max_pool2d():
    """
    default
    """
    np.random.seed(33)
    paddle.seed(33)
    paddle.disable_static()
    x = randtool("float", -10, 10, [1, 1, 3, 3])
    indices = randtool("int", 0, 10, [1, 1, 3, 3]).astype("int32")
    kernel_size = 2
    output_size = [1, 1, 6, 6]
    res = np.array(
        [
            [
                [
                    [-1.78118394, 0.0, 0.0, 0.0, 9.06504063, 0.0],
                    [-1.00049158, -5.02979745, 3.60901609, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            ]
        ]
    )
    exp = paddle.nn.functional.max_unpool2d(
        x=paddle.to_tensor(x), indices=paddle.to_tensor(indices), kernel_size=kernel_size, output_size=output_size
    )
    assert np.allclose(exp.numpy(), res)
