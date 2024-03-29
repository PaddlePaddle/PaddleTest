#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_instancenorm2D
"""
from apibase import APIBase
import paddle
import pytest
import numpy as np


class TestInstanceNorm2D(APIBase):
    """
    test
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float64]
        paddle.set_default_dtype("float64")
        # self.debug = True
        # self.static = True
        # enable check grad
        # self.enable_backward = True


obj = TestInstanceNorm2D(paddle.nn.InstanceNorm2D)


@pytest.mark.api_nn_InstanceNorm2D_parameters
def test_instancenorm2D_base():
    """
    num_features=2
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
            [[1.1251787, -1.0467086, -1.3605211], [0.35687235, 1.2469171, -0.32173783]],
            [[1.7171342, 0.37617627, -0.54773897], [-0.9501806, -1.1719385, 0.5765487]],
        ],
        [
            [[0.47514185, -1.2147375, 0.29438585], [1.8105773, -0.6669844, -0.69838285]],
            [[-1.0330551, -1.030586, -0.11365502], [1.8085632, 0.6913647, -0.32263255]],
        ],
    ]

    obj.base(res=res, num_features=2, data=x_data)


@pytest.mark.api_nn_InstanceNorm2D_parameters
def test_instancenorm2D1():
    """
    num_features=2
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
            [[1.1251787, -1.0467086, -1.3605211], [0.35687235, 1.2469171, -0.32173783]],
            [[1.7171342, 0.37617627, -0.54773897], [-0.9501806, -1.1719385, 0.5765487]],
        ],
        [
            [[0.47514185, -1.2147375, 0.29438585], [1.8105773, -0.6669844, -0.69838285]],
            [[-1.0330551, -1.030586, -0.11365502], [1.8085632, 0.6913647, -0.32263255]],
        ],
    ]

    obj.run(res=res, num_features=2, data=x_data)


@pytest.mark.api_nn_InstanceNorm2D_parameters
def test_instancenorm2D2():
    """
    num_features=2, epsilon<=1e-3
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
            [[1.1251787, -1.0467086, -1.3605211], [0.35687235, 1.2469171, -0.32173783]],
            [[1.7171342, 0.37617627, -0.54773897], [-0.9501806, -1.1719385, 0.5765487]],
        ],
        [
            [[0.47514185, -1.2147375, 0.29438585], [1.8105773, -0.6669844, -0.69838285]],
            [[-1.0330551, -1.030586, -0.11365502], [1.8085632, 0.6913647, -0.32263255]],
        ],
    ]

    obj.run(res=res, num_features=2, data=x_data, epsilon=1e-05)


@pytest.mark.api_nn_InstanceNorm2D_parameters
def test_instancenorm2D3():
    """
    num_features=2, epsilon<=1e-3, momentum=0.1
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
            [[1.1251787, -1.0467086, -1.3605211], [0.35687235, 1.2469171, -0.32173783]],
            [[1.7171342, 0.37617627, -0.54773897], [-0.9501806, -1.1719385, 0.5765487]],
        ],
        [
            [[0.47514185, -1.2147375, 0.29438585], [1.8105773, -0.6669844, -0.69838285]],
            [[-1.0330551, -1.030586, -0.11365502], [1.8085632, 0.6913647, -0.32263255]],
        ],
    ]

    obj.run(res=res, num_features=2, data=x_data, epsilon=1e-05, momentum=0.1)


@pytest.mark.api_nn_InstanceNorm2D_parameters
def test_instancenorm2D4():
    """
    num_features=2, epsilon<=1e-3, momentum=0.9
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
            [[1.1251787, -1.0467086, -1.3605211], [0.35687235, 1.2469171, -0.32173783]],
            [[1.7171342, 0.37617627, -0.54773897], [-0.9501806, -1.1719385, 0.5765487]],
        ],
        [
            [[0.47514185, -1.2147375, 0.29438585], [1.8105773, -0.6669844, -0.69838285]],
            [[-1.0330551, -1.030586, -0.11365502], [1.8085632, 0.6913647, -0.32263255]],
        ],
    ]

    obj.run(res=res, num_features=2, data=x_data, epsilon=1e-05, momentum=0.9)


@pytest.mark.api_nn_InstanceNorm2D_parameters
def test_instancenorm2D5():
    """
    num_features=2, epsilon<=1e-3, momentum=0.9, weight_attr=False, bias_attr=False
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
            [[1.1251787, -1.0467086, -1.3605211], [0.35687235, 1.2469171, -0.32173783]],
            [[1.7171342, 0.37617627, -0.54773897], [-0.9501806, -1.1719385, 0.5765487]],
        ],
        [
            [[0.47514185, -1.2147375, 0.29438585], [1.8105773, -0.6669844, -0.69838285]],
            [[-1.0330551, -1.030586, -0.11365502], [1.8085632, 0.6913647, -0.32263255]],
        ],
    ]

    obj.run(res=res, num_features=2, data=x_data, epsilon=1e-05, momentum=0.9)


@pytest.mark.api_nn_InstanceNorm2D_parameters
def test_instancenorm2D6():
    """
    num_features=2, epsilon<=1e-3, momentum=0.9, weight_attr=False, bias_attr=False, data_format='NCHW'
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
            [[1.1251787, -1.0467086, -1.3605211], [0.35687235, 1.2469171, -0.32173783]],
            [[1.7171342, 0.37617627, -0.54773897], [-0.9501806, -1.1719385, 0.5765487]],
        ],
        [
            [[0.47514185, -1.2147375, 0.29438585], [1.8105773, -0.6669844, -0.69838285]],
            [[-1.0330551, -1.030586, -0.11365502], [1.8085632, 0.6913647, -0.32263255]],
        ],
    ]

    obj.run(
        res=res,
        num_features=2,
        data=x_data,
        epsilon=1e-05,
        momentum=0.9,
        weight_attr=False,
        bias_attr=False,
        data_format="NCHW",
    )


@pytest.mark.api_nn_InstanceNorm2D_parameters
def test_instancenorm2D7():
    """
    num_features=2, epsilon<=1e-3, momentum=0.9, weight_attr=False, bias_attr=False, data_format='NCW'
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
            [[1.1251787, -1.0467086, -1.3605211], [0.35687235, 1.2469171, -0.32173783]],
            [[1.7171342, 0.37617627, -0.54773897], [-0.9501806, -1.1719385, 0.5765487]],
        ],
        [
            [[0.47514185, -1.2147375, 0.29438585], [1.8105773, -0.6669844, -0.69838285]],
            [[-1.0330551, -1.030586, -0.11365502], [1.8085632, 0.6913647, -0.32263255]],
        ],
    ]
    obj.run(res=res, num_features=2, data=x_data, epsilon=1e-05, momentum=0.9, data_format="NCHW")
