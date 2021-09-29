#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_functional_elu.py
"""
from apibase import APIBase
from apibase import randtool
import paddle
import pytest
import numpy as np


class TestFunctionalPad(APIBase):
    """
    test
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32, np.float64]
        # self.types = [np.float16, np.float32, np.float64]
        # self.debug = True
        # self.static = True
        # enable check grad
        # self.enable_backward = True


obj = TestFunctionalPad(paddle.nn.functional.pad)


@pytest.mark.api_nn_functional_pad_vartype
def test_pad_base():
    """
    x = randtool("float", -10, 10, [3, 3, 3])
    pad = [1, 1]
    mode = "constant"
    value = 0.0
    data_format = "NCL"
    """
    x = randtool("float", -10, 10, [3, 3, 3])
    pad = [1, 1]
    mode = "constant"
    value = 0.0
    data_format = "NCL"
    res = np.array(
        [
            [
                [0.0, -5.0297976, -1.0004916, -1.781184, 0.0],
                [0.0, -4.7940063, 7.4079137, -6.2992015, 0.0],
                [0.0, -9.606771, 9.065041, 3.6090162, 0.0],
            ],
            [
                [0.0, -0.26823747, 9.300536, -2.1320252, 0.0],
                [0.0, -8.408849, -2.9718516, -6.727297, 0.0],
                [0.0, 9.663337, 7.6125636, -0.11873063, 0.0],
            ],
            [
                [0.0, -1.9808152, -0.97417074, 4.4175367, 0.0],
                [0.0, -5.0446343, 2.455599, -7.151024, 0.0],
                [0.0, -5.9764743, -8.375646, 9.069446, 0.0],
            ],
        ]
    )
    obj.base(res=res, x=x, pad=pad, mode=mode, value=value, data_format=data_format)


@pytest.mark.api_nn_functional_pad_parameters
def test_pad():
    """
    x = randtool("float", -10, 10, [3, 2, 1, 2])
    pad = [1, 1, 2, 3]
    mode = "constant"
    value = 2.0
    data_format = "NCHW"
    """
    x = randtool("float", -10, 10, [3, 2, 1, 2])
    pad = [1, 1, 2, 3]
    mode = "constant"
    value = 2.0
    data_format = "NCHW"
    res = np.array(
        [
            [
                [
                    [2.0, 2.0, 2.0, 2.0],
                    [2.0, 2.0, 2.0, 2.0],
                    [2.0, -8.88523461, 1.99072967, 2.0],
                    [2.0, 2.0, 2.0, 2.0],
                    [2.0, 2.0, 2.0, 2.0],
                    [2.0, 2.0, 2.0, 2.0],
                ],
                [
                    [2.0, 2.0, 2.0, 2.0],
                    [2.0, 2.0, 2.0, 2.0],
                    [2.0, 4.45995261, 9.40579439, 2.0],
                    [2.0, 2.0, 2.0, 2.0],
                    [2.0, 2.0, 2.0, 2.0],
                    [2.0, 2.0, 2.0, 2.0],
                ],
            ],
            [
                [
                    [2.0, 2.0, 2.0, 2.0],
                    [2.0, 2.0, 2.0, 2.0],
                    [2.0, 6.43138915, 0.55102135, 2.0],
                    [2.0, 2.0, 2.0, 2.0],
                    [2.0, 2.0, 2.0, 2.0],
                    [2.0, 2.0, 2.0, 2.0],
                ],
                [
                    [2.0, 2.0, 2.0, 2.0],
                    [2.0, 2.0, 2.0, 2.0],
                    [2.0, -3.37046541, -2.92035609, 2.0],
                    [2.0, 2.0, 2.0, 2.0],
                    [2.0, 2.0, 2.0, 2.0],
                    [2.0, 2.0, 2.0, 2.0],
                ],
            ],
            [
                [
                    [2.0, 2.0, 2.0, 2.0],
                    [2.0, 2.0, 2.0, 2.0],
                    [2.0, -8.41939397, 1.11828761, 2.0],
                    [2.0, 2.0, 2.0, 2.0],
                    [2.0, 2.0, 2.0, 2.0],
                    [2.0, 2.0, 2.0, 2.0],
                ],
                [
                    [2.0, 2.0, 2.0, 2.0],
                    [2.0, 2.0, 2.0, 2.0],
                    [2.0, -6.68411074, -4.09524338, 2.0],
                    [2.0, 2.0, 2.0, 2.0],
                    [2.0, 2.0, 2.0, 2.0],
                    [2.0, 2.0, 2.0, 2.0],
                ],
            ],
        ]
    )
    obj.run(res=res, x=x, pad=pad, mode=mode, value=value, data_format=data_format)


@pytest.mark.api_nn_functional_pad_parameters
def test_pad1():
    """
    x = np.array([[[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]]])
    pad = [0, 0, 0, 0, 0, 0, 1, 1, 0, 0]
    mode = "constant"
    value = 0
    """
    x = np.array([[[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]]])
    pad = [0, 0, 0, 0, 0, 0, 1, 1, 0, 0]
    mode = "constant"
    value = 0
    res = np.array([[[[[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [0.0, 0.0, 0.0]]]]])
    # data_format = "NCDHW"
    obj.run(res=res, x=x, pad=pad, mode=mode, value=value)


@pytest.mark.api_nn_functional_pad_parameters
def test_pad2():
    """
    x = np.array([[[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]]])
    pad = (0, 1, 1, 1, 2, 0)
    mode = "constant"
    value = 0
    data_format = "NCDHW"
    """
    x = np.array([[[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]]])
    pad = (0, 1, 1, 1, 2, 0)
    mode = "constant"
    value = 0
    data_format = "NCDHW"
    res = np.array(
        [
            [
                [
                    [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
                    [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
                    [[0.0, 0.0, 0.0, 0.0], [1.0, 2.0, 3.0, 0.0], [4.0, 5.0, 6.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
                ]
            ]
        ]
    )
    obj.run(res=res, x=x, pad=pad, mode=mode, value=value, data_format=data_format)


@pytest.mark.api_nn_functional_pad_parameters
def test_pad3():
    """
    x = np.array([[[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]]])
    pad = [0, 0, 1, 1, 0, 0]
    mode = "constant"
    value = 0
    data_format = "NCDHW"
    """
    x = np.array([[[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]]])
    pad = [0, 0, 1, 1, 0, 0]
    mode = "constant"
    value = 0
    data_format = "NCDHW"
    res = np.array([[[[[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [0.0, 0.0, 0.0]]]]])
    obj.run(res=res, x=x, pad=pad, mode=mode, value=value, data_format=data_format)


@pytest.mark.api_nn_functional_pad_parameters
def test_pad4():
    """
    x = np.array([[[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]]])
    pad = (0, 1, 1, 1, 2, 0)
    mode = "circular"
    value = 0
    data_format = "NCDHW"
    """
    x = np.array([[[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]]])
    pad = (0, 1, 1, 1, 2, 0)
    mode = "circular"
    value = 0
    data_format = "NCDHW"
    res = np.array(
        [
            [
                [
                    [[4.0, 5.0, 6.0, 4.0], [1.0, 2.0, 3.0, 1.0], [4.0, 5.0, 6.0, 4.0], [1.0, 2.0, 3.0, 1.0]],
                    [[4.0, 5.0, 6.0, 4.0], [1.0, 2.0, 3.0, 1.0], [4.0, 5.0, 6.0, 4.0], [1.0, 2.0, 3.0, 1.0]],
                    [[4.0, 5.0, 6.0, 4.0], [1.0, 2.0, 3.0, 1.0], [4.0, 5.0, 6.0, 4.0], [1.0, 2.0, 3.0, 1.0]],
                ]
            ]
        ]
    )
    obj.run(res=res, x=x, pad=pad, mode=mode, value=value, data_format=data_format)


@pytest.mark.api_nn_functional_pad_parameters
def test_pad5():
    """
    x = np.array([[[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]]])
    pad = (0, 1, 1, 1, 2, 0)
    mode = "circular"
    value = 0
    data_format = "NDHWC"
    """
    x = np.array([[[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]]])
    pad = (0, 1, 1, 1, 2, 0)
    mode = "circular"
    value = 0
    data_format = "NDHWC"
    res = np.array(
        [
            [
                [
                    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [1.0, 2.0, 3.0]],
                    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [1.0, 2.0, 3.0]],
                    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [1.0, 2.0, 3.0]],
                ],
                [
                    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [1.0, 2.0, 3.0]],
                    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [1.0, 2.0, 3.0]],
                    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [1.0, 2.0, 3.0]],
                ],
                [
                    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [1.0, 2.0, 3.0]],
                    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [1.0, 2.0, 3.0]],
                    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [1.0, 2.0, 3.0]],
                ],
            ]
        ]
    )
    obj.run(res=res, x=x, pad=pad, mode=mode, value=value, data_format=data_format)


@pytest.mark.api_nn_functional_pad_parameters
def test_pad6():
    """
    x = np.array([[[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]]])
    pad = (2, 1, 3, 0, 2, 0)
    mode = "replicate"
    data_format = "NDHWC"
    """
    x = np.array([[[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]]])
    pad = (2, 1, 3, 0, 2, 0)
    mode = "replicate"
    data_format = "NDHWC"
    res = np.array(
        [
            [
                [
                    [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [4.0, 5.0, 6.0]],
                    [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [4.0, 5.0, 6.0]],
                    [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [4.0, 5.0, 6.0]],
                    [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [4.0, 5.0, 6.0]],
                ],
                [
                    [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [4.0, 5.0, 6.0]],
                    [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [4.0, 5.0, 6.0]],
                    [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [4.0, 5.0, 6.0]],
                    [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [4.0, 5.0, 6.0]],
                ],
                [
                    [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [4.0, 5.0, 6.0]],
                    [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [4.0, 5.0, 6.0]],
                    [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [4.0, 5.0, 6.0]],
                    [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [4.0, 5.0, 6.0]],
                ],
            ]
        ]
    )
    obj.run(res=res, x=x, pad=pad, mode=mode, data_format=data_format)


@pytest.mark.api_nn_functional_pad_parameters
def test_pad7():
    """
    x = np.array([[[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]]])
    pad = (2, 2, 1, 1, 0, 0)
    mode = "reflect"
    data_format = "NCDHW"
    """
    x = np.array([[[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]]])
    pad = (2, 2, 1, 1, 0, 0)
    mode = "reflect"
    data_format = "NCDHW"
    res = np.array(
        [
            [
                [
                    [
                        [6.0, 5.0, 4.0, 5.0, 6.0, 5.0, 4.0],
                        [3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0],
                        [6.0, 5.0, 4.0, 5.0, 6.0, 5.0, 4.0],
                        [3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0],
                    ]
                ]
            ]
        ]
    )
    obj.run(res=res, x=x, pad=pad, mode=mode, data_format=data_format)
