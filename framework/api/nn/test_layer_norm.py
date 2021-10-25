#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test nn.LayerNorm
"""
from apibase import APIBase
import paddle
import pytest
import numpy as np


class TestLayerNorm(APIBase):
    """
    test
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32]
        # self.debug = True
        # static backward has a bug, waitting for fix.
        # self.static = False
        # enable check grad
        # self.no_grad_var = ['padding']
        self.rtol = 1e-6
        self.enable_backward = True


obj = TestLayerNorm(paddle.nn.LayerNorm)


@pytest.mark.api_nn_LayerNorm_vartype
def test_layer_norm_base():
    """
    input=4D, num_features=2
    """
    x_data = np.array(
        [
            [
                [[0.6964692, 0.28613934, 0.22685145], [0.5513148, 0.71946895, 0.42310646]],
                [[0.9807642, 0.6848297, 0.4809319], [0.39211753, 0.343178, 0.7290497]],
            ],
            [
                [[0.43857226, 0.0596779, 0.39804426], [0.7379954, 0.18249173, 0.17545176]],
                [[0.53155136, 0.53182757, 0.63440096], [0.8494318, 0.7244553, 0.6110235]],
            ],
        ]
    )

    res = [
        [
            [[0.7187893, -1.2011795, -1.4785926], [0.03959922, 0.82640713, -0.5602985]],
            [[2.0490303, 0.66432714, -0.28972825], [-0.7052984, -0.93429065, 0.8712362]],
        ],
        [
            [[-0.21512909, -1.8132395, -0.38606915], [1.0477855, -1.2952322, -1.3249255]],
            [[0.17704056, 0.17820556, 0.6108423], [1.5178049, 0.99067575, 0.5122401]],
        ],
    ]

    obj.base(res=res, normalized_shape=(2, 2, 3), data=x_data)


@pytest.mark.api_nn_LayerNorm_parameters
def test_layernorm1():
    """
    input=4D, num_features=2
    """
    x_data = np.array(
        [
            [
                [[0.6964692, 0.28613934, 0.22685145], [0.5513148, 0.71946895, 0.42310646]],
                [[0.9807642, 0.6848297, 0.4809319], [0.39211753, 0.343178, 0.7290497]],
            ],
            [
                [[0.43857226, 0.0596779, 0.39804426], [0.7379954, 0.18249173, 0.17545176]],
                [[0.53155136, 0.53182757, 0.63440096], [0.8494318, 0.7244553, 0.6110235]],
            ],
        ]
    )
    res = [
        [
            [[0.7187893, -1.2011795, -1.4785926], [0.03959922, 0.82640713, -0.5602985]],
            [[2.0490303, 0.66432714, -0.28972825], [-0.7052984, -0.93429065, 0.8712362]],
        ],
        [
            [[-0.21512909, -1.8132395, -0.38606915], [1.0477855, -1.2952322, -1.3249255]],
            [[0.17704056, 0.17820556, 0.6108423], [1.5178049, 0.99067575, 0.5122401]],
        ],
    ]

    obj.run(res=res, normalized_shape=(2, 2, 3), data=x_data)


@pytest.mark.api_nn_LayerNorm_parameters
def test_layernorm2():
    """
    input=4D, num_features=2, epsilon<=1e-03
    """
    x_data = np.array(
        [
            [
                [[0.6964692, 0.28613934, 0.22685145], [0.5513148, 0.71946895, 0.42310646]],
                [[0.9807642, 0.6848297, 0.4809319], [0.39211753, 0.343178, 0.7290497]],
            ],
            [
                [[0.43857226, 0.0596779, 0.39804426], [0.7379954, 0.18249173, 0.17545176]],
                [[0.53155136, 0.53182757, 0.63440096], [0.8494318, 0.7244553, 0.6110235]],
            ],
        ]
    )
    res = [
        [
            [[0.7187893, -1.2011795, -1.4785926], [0.03959922, 0.82640713, -0.5602985]],
            [[2.0490303, 0.66432714, -0.28972825], [-0.7052984, -0.93429065, 0.8712362]],
        ],
        [
            [[-0.21512909, -1.8132395, -0.38606915], [1.0477855, -1.2952322, -1.3249255]],
            [[0.17704056, 0.17820556, 0.6108423], [1.5178049, 0.99067575, 0.5122401]],
        ],
    ]

    obj.run(res=res, normalized_shape=(2, 2, 3), data=x_data, epsilon=1e-5)


@pytest.mark.api_nn_LayerNorm_parameters
def test_layernorm3():
    """
    input=4D, num_features=2, epsilon<=1e-03, weight_attr=False, bias_attr=False
    """
    x_data = np.array(
        [
            [
                [[0.6964692, 0.28613934, 0.22685145], [0.5513148, 0.71946895, 0.42310646]],
                [[0.9807642, 0.6848297, 0.4809319], [0.39211753, 0.343178, 0.7290497]],
            ],
            [
                [[0.43857226, 0.0596779, 0.39804426], [0.7379954, 0.18249173, 0.17545176]],
                [[0.53155136, 0.53182757, 0.63440096], [0.8494318, 0.7244553, 0.6110235]],
            ],
        ]
    )
    res = [
        [
            [[0.7187893, -1.2011795, -1.4785926], [0.03959922, 0.82640713, -0.5602985]],
            [[2.0490303, 0.66432714, -0.28972825], [-0.7052984, -0.93429065, 0.8712362]],
        ],
        [
            [[-0.21512909, -1.8132395, -0.38606915], [1.0477855, -1.2952322, -1.3249255]],
            [[0.17704056, 0.17820556, 0.6108423], [1.5178049, 0.99067575, 0.5122401]],
        ],
    ]

    obj.run(res=res, normalized_shape=(2, 2, 3), data=x_data, epsilon=1e-5, weight_attr=False, bias_attr=False)


@pytest.mark.api_nn_LayerNorm_parameters
def test_layernorm4():
    """
    input=3D, num_features=2
    """
    x_data = np.array(
        [
            [[0.6964692, 0.28613934, 0.22685145], [0.5513148, 0.71946895, 0.42310646]],
            [[0.9807642, 0.6848297, 0.4809319], [0.39211753, 0.343178, 0.7290497]],
        ]
    )
    res = [
        [[1.1251786, -1.0467087, -1.3605212], [0.3568722, 1.2469169, -0.32173792]],
        [[1.7171341, 0.37617615, -0.5477391], [-0.9501807, -1.1719387, 0.5765486]],
    ]

    obj.run(res=res, normalized_shape=(2, 3), data=x_data)


@pytest.mark.api_nn_LayerNorm_parameters
def test_layernorm5():
    """
    input=3D, num_features=2, epsilon<=1e-03,
    """
    x_data = np.array(
        [
            [[0.6964692, 0.28613934, 0.22685145], [0.5513148, 0.71946895, 0.42310646]],
            [[0.9807642, 0.6848297, 0.4809319], [0.39211753, 0.343178, 0.7290497]],
        ]
    )
    res = [
        [[1.1251786, -1.0467087, -1.3605212], [0.3568722, 1.2469169, -0.32173792]],
        [[1.7171341, 0.37617615, -0.5477391], [-0.9501807, -1.1719387, 0.5765486]],
    ]

    obj.run(res=res, normalized_shape=(2, 3), data=x_data, epsilon=1e-5)


@pytest.mark.api_nn_LayerNorm_parameters
def test_layernorm6():
    """
    input=3D, num_features=2, epsilon<=1e-03, weight_attr=False, bias_attr=False
    """
    x_data = np.array(
        [
            [[0.6964692, 0.28613934, 0.22685145], [0.5513148, 0.71946895, 0.42310646]],
            [[0.9807642, 0.6848297, 0.4809319], [0.39211753, 0.343178, 0.7290497]],
        ]
    )
    res = [
        [[1.1251786, -1.0467087, -1.3605212], [0.3568722, 1.2469169, -0.32173792]],
        [[1.7171341, 0.37617615, -0.5477391], [-0.9501807, -1.1719387, 0.5765486]],
    ]

    obj.run(res=res, normalized_shape=(2, 3), data=x_data, epsilon=1e-5, weight_attr=False, bias_attr=False)


@pytest.mark.api_nn_LayerNorm_parameters
def test_layernorm7():
    """
    input=2D, num_features=2
    """
    x_data = np.array([[0.6964692, 0.28613934, 0.22685145], [0.5513148, 0.71946895, 0.42310646]])
    res = [[1.4045198, -0.5603123, -0.84420764], [-0.10968406, 1.275481, -1.1657962]]

    obj.run(res=res, normalized_shape=(3,), data=x_data)


@pytest.mark.api_nn_LayerNorm_parameters
def test_layernorm8():
    """
    input=2D, num_features=2, epsilon<=1e-03
    """
    x_data = np.array([[0.6964692, 0.28613934, 0.22685145], [0.5513148, 0.71946895, 0.42310646]])
    res = [[1.4045198, -0.5603123, -0.84420764], [-0.10968406, 1.275481, -1.1657962]]

    obj.run(res=res, normalized_shape=(3,), data=x_data, epsilon=1e-5)


@pytest.mark.api_nn_LayerNorm_parameters
def test_layernorm9():
    """
    input=2D, num_features=2, epsilon<=1e-03, weight_attr=False, bias_attr=False
    """
    x_data = np.array([[0.6964692, 0.28613934, 0.22685145], [0.5513148, 0.71946895, 0.42310646]])
    res = [[1.4045198, -0.5603123, -0.84420764], [-0.10968406, 1.275481, -1.1657962]]

    obj.run(res=res, normalized_shape=(3,), data=x_data, epsilon=1e-5, weight_attr=False, bias_attr=False)
