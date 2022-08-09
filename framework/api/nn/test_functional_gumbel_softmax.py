#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_functional_gumbel_softmax
"""
import sys
from apibase import APIBase
import paddle
import pytest
import numpy as np

sys.path.append("../..")
from utils.interceptor import skip_not_compile_gpu


class TestFunctionalGumbelSoftmax(APIBase):
    """
    test
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32, np.float64]
        # self.debug = True
        # self.static = True
        # enable check grad
        # self.enable_backward = True


# cpu
obj = TestFunctionalGumbelSoftmax(paddle.nn.functional.gumbel_softmax)
obj.places = [paddle.CPUPlace()]


@pytest.mark.api_nn_gumbel_softmax_vartype
def test_functional_gumbel_softmax_base():
    """
    base
    """
    obj.delta = 1e-5
    x = np.array([6.0, 7.0, 8.0, 9.0])
    res = np.array([0.01416574, 0.83565587, 0.01048952, 0.13968873])
    obj.base(res=res, x=x)


@pytest.mark.api_nn_gumbel_softmax_parameters
def test_functional_gumbel_softmax0():
    """
    default
    """
    obj.delta = 1e-7
    x = np.array([[2.0, 3.0, 4.0, 5.0], [3.0, 4.0, 5.0, 6.0], [7.0, 8.0, 8.0, 9.0]])
    res = np.array(
        [
            [0.01416553, 0.83565838, 0.01048938, 0.13968671],
            [0.00625133, 0.02044939, 0.79816702, 0.17513226],
            [0.00557371, 0.05935366, 0.00904187, 0.92603076],
        ]
    )
    obj.run(res=res, x=x)


@pytest.mark.api_nn_gumbel_softmax_parameters
def test_functional_gumbel_softmax1():
    """
    x: multi_dim
    """
    x = np.array(
        [
            [[2.0, 3.0, 4.0, 5.0], [3.0, 4.0, 5.0, 6.0], [7.0, 8.0, 8.0, 9.0]],
            [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [6.0, 7.0, 8.0, 9.0]],
        ]
    )
    res = np.array(
        [
            [
                [0.01416553, 0.83565838, 0.01048938, 0.13968671],
                [0.00625133, 0.02044939, 0.79816702, 0.17513226],
                [0.00557371, 0.05935366, 0.00904187, 0.92603076],
            ],
            [
                [0.16894336, 0.07674793, 0.36803119, 0.38627751],
                [0.24528688, 0.04746089, 0.12632742, 0.58092482],
                [0.00059851, 0.05744077, 0.84302186, 0.09893887],
            ],
        ]
    )
    obj.run(res=res, x=x)


@pytest.mark.api_nn_gumbel_softmax_parameters
def test_functional_gumbel_softmax2():
    """
    t = 4.0
    """
    x = np.array(
        [
            [[2.0, 3.0, 4.0, 5.0], [3.0, 4.0, 5.0, 6.0], [7.0, 8.0, 8.0, 9.0]],
            [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [6.0, 7.0, 8.0, 9.0]],
        ]
    )
    res = np.array(
        [
            [
                [0.15453308, 0.42827273, 0.14335097, 0.27384322],
                [0.12489108, 0.16796098, 0.41981868, 0.28732926],
                [0.13288625, 0.24005229, 0.14997144, 0.47709002],
            ],
            [
                [0.23443704, 0.19246759, 0.28481451, 0.28828086],
                [0.26660162, 0.17681869, 0.22584923, 0.33073046],
                [0.07224468, 0.22612199, 0.44258578, 0.25904755],
            ],
        ]
    )
    obj.run(res=res, x=x, temperature=4.0)


@pytest.mark.api_nn_gumbel_softmax_parameters
def test_functional_gumbel_softmax3():
    """
    t = 4.0
    hard = True
    """
    x = np.array(
        [
            [[2.0, 3.0, 4.0, 5.0], [3.0, 4.0, 5.0, 6.0], [7.0, 8.0, 8.0, 9.0]],
            [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [6.0, 7.0, 8.0, 9.0]],
        ]
    )
    res = np.array(
        [
            [[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0]],
        ]
    )
    obj.run(res=res, x=x, temperature=4.0, hard=True)


@pytest.mark.api_nn_gumbel_softmax_parameters
def test_functional_gumbel_softmax4():
    """
    t = 4.0
    hard = True
    axis = 0
    """
    x = np.array(
        [
            [
                [1.0, 3.0, 2.0, 3.0, 3.0],
                [0.0, -2.0, 3.0, 2.0, 2.0],
                [-2.0, -2.0, 0.0, 1.0, 0.0],
                [3.0, 2.0, 1.0, 1.0, 0.0],
            ],
            [
                [2.0, -1.0, 1.0, 2.0, -1.0],
                [2.0, -2.0, 2.0, 2.0, 3.0],
                [3.0, 0.0, -1.0, 1.0, 0.0],
                [-1.0, -2.0, 3.0, -2.0, -2.0],
            ],
            [
                [-2.0, 0.0, -2.0, -2.0, 0.0],
                [0.0, -2.0, 2.0, -1.0, 0.0],
                [1.0, 3.0, 0.0, -1.0, 3.0],
                [3.0, 0.0, -2.0, 1.0, 1.0],
            ],
        ]
    )
    res = np.array(
        [
            [
                [1.0, 1.0, 0.0, 1.0, 1.0],
                [0.0, 1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 1.0, 0.0],
            ],
            [
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 1.0, 1.0],
                [1.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
            ],
        ]
    )
    obj.run(res=res, x=x, temperature=4.0, hard=True, axis=0)


@skip_not_compile_gpu
@pytest.mark.api_nn_gumbel_softmax_parameters
def test_functional_gumbel_softmax5():
    """
    t = 4.0
    hard = True
    axis = 0
    """
    # gpu
    obj1 = TestFunctionalGumbelSoftmax(paddle.nn.functional.gumbel_softmax)
    obj1.places = [paddle.CUDAPlace(0)]
    x = np.array(
        [
            [
                [1.0, 3.0, 2.0, 3.0, 3.0],
                [0.0, -2.0, 3.0, 2.0, 2.0],
                [-2.0, -2.0, 0.0, 1.0, 0.0],
                [3.0, 2.0, 1.0, 1.0, 0.0],
            ],
            [
                [2.0, -1.0, 1.0, 2.0, -1.0],
                [2.0, -2.0, 2.0, 2.0, 3.0],
                [3.0, 0.0, -1.0, 1.0, 0.0],
                [-1.0, -2.0, 3.0, -2.0, -2.0],
            ],
            [
                [-2.0, 0.0, -2.0, -2.0, 0.0],
                [0.0, -2.0, 2.0, -1.0, 0.0],
                [1.0, 3.0, 0.0, -1.0, 3.0],
                [3.0, 0.0, -2.0, 1.0, 1.0],
            ],
        ]
    )
    res = np.array(
        [
            [
                [0.0, 1.0, 1.0, 1.0, 1.0],
                [0.0, 1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0, 1.0],
            ],
            [
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 1.0, 1.0],
                [1.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
            ],
        ]
    )
    obj1.run(res=res, x=x, temperature=4.0, hard=True, axis=0)
