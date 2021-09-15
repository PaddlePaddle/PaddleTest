#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_GroupNorm
"""
from apibase import APIBase
from apibase import randtool
import paddle
import pytest
import numpy as np


def group_norm_naive_for_general_dimension(x, scale, bias, epsilon, groups):
    """
    naive GroupNorm
    """
    input_shape = x.shape
    N = x.shape[0]
    G = groups
    x = x.reshape((N * G, -1))
    mean = np.mean(x, axis=1, keepdims=True)
    var = np.var(x, axis=1, keepdims=True)
    output = (x - mean) / np.sqrt(var + epsilon)
    output = output.reshape(input_shape) * scale.reshape((-1, 1, 1)) + bias.reshape((-1, 1, 1))
    return output


class TestGroupNorm(APIBase):
    """
    test
    """

    def hook(self):
        """
        implement
        """
        # should support fp64, fp64 waitting for fix.
        self.types = [np.float32]
        self.delta = 1e-3 * 5
        self.rtol = 1e-3
        # self.debug = True
        # self.static = True
        # enable check grad
        self.enable_backward = True


obj = TestGroupNorm(paddle.nn.GroupNorm)


@pytest.mark.api_nn_GroupNorm_vartype
def test_group_norm_base():
    """
    input_shape=(2,2,2,3)
    """
    x_data = np.array(
        [
            [
                [[0.6964692, 0.28613934], [0.22685145, 0.5513148], [0.71946895, 0.42310646]],
                [[0.9807642, 0.6848297], [0.4809319, 0.39211753], [0.343178, 0.7290497]],
            ],
            [
                [[0.43857226, 0.0596779], [0.39804426, 0.7379954], [0.18249173, 0.17545176]],
                [[0.53155136, 0.53182757], [0.63440096, 0.8494318], [0.7244553, 0.6110235]],
            ],
        ]
    )

    res = [
        [
            [[1.1251787, -1.0467088], [-1.3605214, 0.35687226], [1.2469171, -0.32173797]],
            [[1.7171334, 0.37617597], [-0.54773885, -0.9501803], [-1.1719382, 0.57654834]],
        ],
        [
            [[0.47514182, -1.2147375], [0.29438582, 1.8105773], [-0.6669844, -0.6983829]],
            [[-1.0330558, -1.0305867], [-0.11365522, 1.8085641], [0.69136494, -0.32263285]],
        ],
    ]

    obj.base(res=res, num_channels=2, num_groups=2, data=x_data)


@pytest.mark.api_nn_GroupNorm_parameters
def test_group_norm1():
    """
    input_shape=(2,2,2,3)
    """
    x_data = np.array(
        [
            [
                [[0.6964692, 0.28613934], [0.22685145, 0.5513148], [0.71946895, 0.42310646]],
                [[0.9807642, 0.6848297], [0.4809319, 0.39211753], [0.343178, 0.7290497]],
            ],
            [
                [[0.43857226, 0.0596779], [0.39804426, 0.7379954], [0.18249173, 0.17545176]],
                [[0.53155136, 0.53182757], [0.63440096, 0.8494318], [0.7244553, 0.6110235]],
            ],
        ]
    )

    res = [
        [
            [[1.1251787, -1.0467088], [-1.3605214, 0.35687226], [1.2469171, -0.32173797]],
            [[1.7171334, 0.37617597], [-0.54773885, -0.9501803], [-1.1719382, 0.57654834]],
        ],
        [
            [[0.47514182, -1.2147375], [0.29438582, 1.8105773], [-0.6669844, -0.6983829]],
            [[-1.0330558, -1.0305867], [-0.11365522, 1.8085641], [0.69136494, -0.32263285]],
        ],
    ]

    obj.run(res=res, num_channels=2, num_groups=2, data=x_data)


@pytest.mark.api_nn_GroupNorm_parameters
def test_group_norm2():
    """
    input_shape=(2,2,2,3), epsilon<=1e-03
    """
    x_data = np.array(
        [
            [
                [[0.6964692, 0.28613934], [0.22685145, 0.5513148], [0.71946895, 0.42310646]],
                [[0.9807642, 0.6848297], [0.4809319, 0.39211753], [0.343178, 0.7290497]],
            ],
            [
                [[0.43857226, 0.0596779], [0.39804426, 0.7379954], [0.18249173, 0.17545176]],
                [[0.53155136, 0.53182757], [0.63440096, 0.8494318], [0.7244553, 0.6110235]],
            ],
        ]
    )

    res = [
        [
            [[1.1251787, -1.0467088], [-1.3605214, 0.35687226], [1.2469171, -0.32173797]],
            [[1.7171334, 0.37617597], [-0.54773885, -0.9501803], [-1.1719382, 0.57654834]],
        ],
        [
            [[0.47514182, -1.2147375], [0.29438582, 1.8105773], [-0.6669844, -0.6983829]],
            [[-1.0330558, -1.0305867], [-0.11365522, 1.8085641], [0.69136494, -0.32263285]],
        ],
    ]

    obj.run(res=res, num_channels=2, num_groups=2, data=x_data, epsilon=1e-05)


@pytest.mark.api_nn_GroupNorm_parameters
def test_group_norm3():
    """
    input_shape=(2,2,2,3), epsilon<=1e-03, weight=False, bias_attr=False
    """
    x_data = np.array(
        [
            [
                [[0.6964692, 0.28613934], [0.22685145, 0.5513148], [0.71946895, 0.42310646]],
                [[0.9807642, 0.6848297], [0.4809319, 0.39211753], [0.343178, 0.7290497]],
            ],
            [
                [[0.43857226, 0.0596779], [0.39804426, 0.7379954], [0.18249173, 0.17545176]],
                [[0.53155136, 0.53182757], [0.63440096, 0.8494318], [0.7244553, 0.6110235]],
            ],
        ]
    )

    res = [
        [
            [[1.1251787, -1.0467088], [-1.3605214, 0.35687226], [1.2469171, -0.32173797]],
            [[1.7171334, 0.37617597], [-0.54773885, -0.9501803], [-1.1719382, 0.57654834]],
        ],
        [
            [[0.47514182, -1.2147375], [0.29438582, 1.8105773], [-0.6669844, -0.6983829]],
            [[-1.0330558, -1.0305867], [-0.11365522, 1.8085641], [0.69136494, -0.32263285]],
        ],
    ]

    obj.run(res=res, num_channels=2, num_groups=2, data=x_data, epsilon=1e-05)


@pytest.mark.api_nn_GroupNorm_parameters
def test_group_norm5():
    """
    input_shape=(2,2,2,3), epsilon<=1e-03, weight=False, bias_attr=False, data_format='NCHW'
    """
    x_data = np.array(
        [
            [
                [[0.6964692, 0.28613934], [0.22685145, 0.5513148], [0.71946895, 0.42310646]],
                [[0.9807642, 0.6848297], [0.4809319, 0.39211753], [0.343178, 0.7290497]],
            ],
            [
                [[0.43857226, 0.0596779], [0.39804426, 0.7379954], [0.18249173, 0.17545176]],
                [[0.53155136, 0.53182757], [0.63440096, 0.8494318], [0.7244553, 0.6110235]],
            ],
        ]
    )

    res = [
        [
            [[1.1251787, -1.0467088], [-1.3605214, 0.35687226], [1.2469171, -0.32173797]],
            [[1.7171334, 0.37617597], [-0.54773885, -0.9501803], [-1.1719382, 0.57654834]],
        ],
        [
            [[0.47514182, -1.2147375], [0.29438582, 1.8105773], [-0.6669844, -0.6983829]],
            [[-1.0330558, -1.0305867], [-0.11365522, 1.8085641], [0.69136494, -0.32263285]],
        ],
    ]

    obj.run(res=res, num_channels=2, num_groups=2, data=x_data, epsilon=1e-05, data_format="NCHW")


@pytest.mark.api_nn_GroupNorm_parameters
def test_group_norm6():
    """
    input_shape=(2, 6, 3, 4, 4, 2), epsilon=1e-5, weight=None, bias_attr=None, data_format='NCHW'
    """
    x_data = randtool("float", -1, 1, [2, 6, 3, 4, 4, 2])
    weight_attr = np.array([1]).astype("float32")
    bias_attr = np.array([0]).astype("float32")
    epsilon = 1e-5
    num_groups = 6
    res = group_norm_naive_for_general_dimension(
        x=x_data, scale=weight_attr, bias=bias_attr, epsilon=epsilon, groups=num_groups
    )

    obj.run(
        res=res,
        num_channels=6,
        num_groups=num_groups,
        weight_attr=None,
        bias_attr=None,
        data=x_data,
        epsilon=epsilon,
        data_format="NCHW",
    )


@pytest.mark.api_nn_GroupNorm_parameters
def test_group_norm7():
    """
    input_shape=(2, 14, 5, 1, 1, 2, 1, 1, 2), epsilon=1e-6, weight=None, bias_attr=None, data_format='NCHW'
    """
    x_data = randtool("float", -1, 1, [2, 14, 5, 1, 1, 2, 1, 1, 2])
    weight_attr = np.array([1]).astype("float32")
    bias_attr = np.array([0]).astype("float32")
    epsilon = 1e-6
    num_groups = 7
    res = group_norm_naive_for_general_dimension(
        x=x_data, scale=weight_attr, bias=bias_attr, epsilon=epsilon, groups=num_groups
    )

    obj.run(
        res=res,
        num_channels=14,
        num_groups=num_groups,
        weight_attr=None,
        bias_attr=None,
        data=x_data,
        epsilon=epsilon,
        data_format="NCHW",
    )
