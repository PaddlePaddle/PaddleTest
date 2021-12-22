#!/usr/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python

"""
test_randint_like
"""

from apibase import APIBase
import paddle
import pytest
import numpy as np


class TestRandintLike(APIBase):
    """
    test randint_like
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32, np.float64, np.float16, np.float32, np.float64]
        # self.debug = True
        # enable check grad
        self.enable_backward = False


obj = TestRandintLike(paddle.randint_like)


@pytest.mark.api_nn_randint_like_vartype
def test_randint_like_base():
    """
    base
    """
    x = np.zeros((4,))
    res = np.array([3.0, 2.0, 2.0, -4.0])
    obj.base(res=res, x=x, low=-4, high=4)


@pytest.mark.api_nn_randint_like_parameters
def test_randint_like0():
    """
    default
    """
    x = np.zeros((4, 5))
    res = np.array(
        [
            [3.0, 2.0, 2.0, -4.0, 0.0],
            [-4.0, -2.0, 3.0, 2.0, 1.0],
            [-2.0, -2.0, 1.0, 1.0, 0.0],
            [3.0, 0.0, -2.0, -1.0, -1.0],
        ]
    )
    obj.run(res=res, x=x, low=-4, high=4)


@pytest.mark.api_nn_randint_like_parameters
def test_randint_like1():
    """
    x: 3-d tensor
    """
    x = np.zeros((2, 4, 5))
    res = np.array(
        [
            [
                [3.0, 2.0, 2.0, -4.0, 0.0],
                [-4.0, -2.0, 3.0, 2.0, 1.0],
                [-2.0, -2.0, 1.0, 1.0, 0.0],
                [3.0, 0.0, -2.0, -1.0, -1.0],
            ],
            [
                [-2.0, 1.0, 1.0, 2.0, -3.0],
                [-4.0, -1.0, -4.0, 1.0, -3.0],
                [-4.0, 3.0, -4.0, 1.0, -4.0],
                [-1.0, 0.0, -4.0, -3.0, -1.0],
            ],
        ]
    )
    obj.run(res=res, x=x, low=-4, high=4)


@pytest.mark.api_nn_randint_like_parameters
def test_randint_like2():
    """
    x: 4-d tensor
    """
    x = np.zeros((2, 2, 4, 5))
    res = np.array(
        [
            [
                [
                    [3.0, 2.0, 2.0, -4.0, 0.0],
                    [-4.0, -2.0, 3.0, 2.0, 1.0],
                    [-2.0, -2.0, 1.0, 1.0, 0.0],
                    [3.0, 0.0, -2.0, -1.0, -1.0],
                ],
                [
                    [-2.0, 1.0, 1.0, 2.0, -3.0],
                    [-4.0, -1.0, -4.0, 1.0, -3.0],
                    [-4.0, 3.0, -4.0, 1.0, -4.0],
                    [-1.0, 0.0, -4.0, -3.0, -1.0],
                ],
            ],
            [
                [
                    [3.0, -4.0, -2.0, 1.0, -1.0],
                    [2.0, -2.0, -1.0, 2.0, 0.0],
                    [-1.0, -2.0, 2.0, 0.0, -3.0],
                    [-2.0, -1.0, 0.0, -2.0, 2.0],
                ],
                [
                    [0.0, -2.0, 3.0, 3.0, 1.0],
                    [-1.0, 0.0, 3.0, 0.0, -2.0],
                    [1.0, 0.0, -4.0, 3.0, 2.0],
                    [-4.0, -1.0, -1.0, 1.0, -3.0],
                ],
            ],
        ]
    )
    obj.run(res=res, x=x, low=-4, high=4)


@pytest.mark.api_nn_randint_like_parameters
def test_randint_like0():
    """
    high=None
    """
    x = np.zeros((4, 5))
    res = np.array(
        [[3.0, 2.0, 2.0, 0.0, 0.0], [0.0, 2.0, 3.0, 2.0, 1.0], [2.0, 2.0, 1.0, 1.0, 0.0], [3.0, 0.0, 2.0, 3.0, 3.0]]
    )
    obj.run(res=res, x=x, low=4)
