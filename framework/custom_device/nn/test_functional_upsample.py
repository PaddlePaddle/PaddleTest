#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_functional_upsample
"""
from apibase import APIBase
from apibase import randtool
import paddle
import pytest
import numpy as np


class TestFunctionalUpsample(APIBase):
    """
    test
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32, np.float64]
        self.debug = True
        self.static = False
        # enable check grad
        self.enable_backward = False


obj = TestFunctionalUpsample(paddle.nn.functional.upsample)


@pytest.mark.api_nn_functional_upsample_vartype
def test_func_upsample_base():
    """
    base
    """
    x = randtool("float", -10, 10, [2, 1, 4, 4])
    size = np.array([8, 8]).astype(np.int32)
    scale_factor = None
    mode = "nearest"
    align_corners = False
    align_mode = 0
    data_format = "NCHW"
    res = np.array(
        [
            [
                [
                    [
                        -5.02979745,
                        -5.02979745,
                        -1.00049158,
                        -1.00049158,
                        -1.78118394,
                        -1.78118394,
                        -4.79400618,
                        -4.79400618,
                    ],
                    [
                        -5.02979745,
                        -5.02979745,
                        -1.00049158,
                        -1.00049158,
                        -1.78118394,
                        -1.78118394,
                        -4.79400618,
                        -4.79400618,
                    ],
                    [
                        7.40791377,
                        7.40791377,
                        -6.29920146,
                        -6.29920146,
                        -9.60677149,
                        -9.60677149,
                        9.06504063,
                        9.06504063,
                    ],
                    [
                        7.40791377,
                        7.40791377,
                        -6.29920146,
                        -6.29920146,
                        -9.60677149,
                        -9.60677149,
                        9.06504063,
                        9.06504063,
                    ],
                    [
                        3.60901609,
                        3.60901609,
                        -0.26823747,
                        -0.26823747,
                        9.30053640,
                        9.30053640,
                        -2.13202522,
                        -2.13202522,
                    ],
                    [
                        3.60901609,
                        3.60901609,
                        -0.26823747,
                        -0.26823747,
                        9.30053640,
                        9.30053640,
                        -2.13202522,
                        -2.13202522,
                    ],
                    [
                        -8.40884857,
                        -8.40884857,
                        -2.97185151,
                        -2.97185151,
                        -6.72729675,
                        -6.72729675,
                        9.66333642,
                        9.66333642,
                    ],
                    [
                        -8.40884857,
                        -8.40884857,
                        -2.97185151,
                        -2.97185151,
                        -6.72729675,
                        -6.72729675,
                        9.66333642,
                        9.66333642,
                    ],
                ]
            ],
            [
                [
                    [
                        7.61256368,
                        7.61256368,
                        -0.11873064,
                        -0.11873064,
                        -1.98081518,
                        -1.98081518,
                        -0.97417074,
                        -0.97417074,
                    ],
                    [
                        7.61256368,
                        7.61256368,
                        -0.11873064,
                        -0.11873064,
                        -1.98081518,
                        -1.98081518,
                        -0.97417074,
                        -0.97417074,
                    ],
                    [
                        4.41753697,
                        4.41753697,
                        -5.04463431,
                        -5.04463431,
                        2.45559905,
                        2.45559905,
                        -7.15102368,
                        -7.15102368,
                    ],
                    [
                        4.41753697,
                        4.41753697,
                        -5.04463431,
                        -5.04463431,
                        2.45559905,
                        2.45559905,
                        -7.15102368,
                        -7.15102368,
                    ],
                    [
                        -5.97647435,
                        -5.97647435,
                        -8.37564541,
                        -8.37564541,
                        9.06944590,
                        9.06944590,
                        -8.88523461,
                        -8.88523461,
                    ],
                    [
                        -5.97647435,
                        -5.97647435,
                        -8.37564541,
                        -8.37564541,
                        9.06944590,
                        9.06944590,
                        -8.88523461,
                        -8.88523461,
                    ],
                    [1.99072967, 1.99072967, 4.45995261, 4.45995261, 9.40579439, 9.40579439, 6.43138915, 6.43138915],
                    [1.99072967, 1.99072967, 4.45995261, 4.45995261, 9.40579439, 9.40579439, 6.43138915, 6.43138915],
                ]
            ],
        ]
    )
    obj.base(
        res=res,
        x=x,
        size=size,
        scale_factor=scale_factor,
        mode=mode,
        align_corners=align_corners,
        align_mode=align_mode,
        data_format=data_format,
    )


@pytest.mark.api_nn_functional_upsample_parameters
def test_func_upsample1():
    """
    default
    """
    x = randtool("float", -10, 10, [1, 2, 2, 1])
    size = np.array([4, 4]).astype(np.int32)
    scale_factor = None
    mode = "bilinear"
    align_corners = True
    align_mode = 1
    data_format = "NHWC"
    res = np.array(
        [
            [
                [[0.55102135], [-0.75614096], [-2.06330323], [-3.37046541]],
                [[-0.60610451], [-2.08855022], [-3.57099596], [-5.05344155]],
                [[-1.76323034], [-3.42095952], [-5.0786888], [-6.73641789]],
                [[-2.92035609], [-4.75336868], [-6.58638146], [-8.41939397]],
            ]
        ]
    )
    obj.run(
        res=res,
        x=x,
        size=size,
        scale_factor=scale_factor,
        mode=mode,
        align_corners=align_corners,
        align_mode=align_mode,
        data_format=data_format,
    )


@pytest.mark.api_nn_functional_upsample_parameters
def test_func_upsample2():
    """
    exception return_mask=True
    """
    x = randtool("float", -10, 10, [1, 2, 2, 2, 1])
    size = np.array([4, 4, 4]).astype(np.int32)
    scale_factor = None
    mode = "trilinear"
    align_corners = True
    align_mode = 1
    data_format = "NDHWC"
    res = np.array(
        [
            [
                [
                    [[1.11828761], [-1.48251195], [-4.08331145], [-6.68411074]],
                    [[-0.61955614], [-1.14148751], [-1.66341892], [-2.18535026]],
                    [[-2.35739982], [-0.80046316], [0.75647337], [2.31340982]],
                    [[-4.09524338], [-0.45943879], [3.17636556], [6.8121697]],
                ],
                [
                    [[-0.17072827], [-1.95292348], [-3.73511869], [-5.51731374]],
                    [[-0.55631344], [-1.4392233], [-2.3221332], [-3.205043]],
                    [[-0.94189862], [-0.92552324], [-0.90914793], [-0.89277259]],
                    [[-1.32748375], [-0.41182317], [0.50383732], [1.41949778]],
                ],
                [
                    [[-1.45974408], [-2.42333509], [-3.38692618], [-4.35051714]],
                    [[-0.49307077], [-1.73695916], [-2.98084758], [-4.22473587]],
                    [[0.47360245], [-1.05058338], [-2.57476918], [-4.09895486]],
                    [[1.44027563], [-0.36420759], [-2.16869072], [-3.97317373]],
                ],
                [
                    [[-2.74875981], [-2.89374661], [-3.03873358], [-3.18372045]],
                    [[-0.4298281], [-2.03469496], [-3.63956185], [-5.24442859]],
                    [[1.88910346], [-1.17564348], [-4.24039031], [-7.30513692]],
                    [[4.20803488], [-0.31659199], [-4.84121862], [-9.36584497]],
                ],
            ]
        ]
    )
    obj.run(
        res=res,
        x=x,
        size=size,
        scale_factor=scale_factor,
        mode=mode,
        align_corners=align_corners,
        align_mode=align_mode,
        data_format=data_format,
    )


@pytest.mark.api_nn_functional_upsample_parameters
def test_func_upsample3():
    """
    default
    """
    x = randtool("float", -10, 10, [1, 2, 2, 1])
    size = np.array([4, 4]).astype(np.int32)
    scale_factor = None
    mode = "bicubic"
    align_corners = False
    align_mode = 0
    data_format = "NHWC"
    res = np.array(
        [
            [
                [[-1.90953963], [0.36574692], [4.11327769], [6.38856424]],
                [[0.64225691], [1.17311928], [2.04748084], [2.57834322]],
                [[4.84521591], [2.50290906], [-1.35500809], [-3.69731493]],
                [[7.39701244], [3.31028143], [-3.42080494], [-7.50753595]],
            ]
        ]
    )
    obj.run(
        res=res,
        x=x,
        size=size,
        scale_factor=scale_factor,
        mode=mode,
        align_corners=align_corners,
        align_mode=align_mode,
        data_format=data_format,
    )


@pytest.mark.api_nn_functional_upsample_parameters
def test_func_upsample4():
    """
    default
    """
    x = randtool("float", -10, 10, [1, 2, 1])
    size = np.array([4]).astype(np.int32)
    scale_factor = None
    mode = "linear"
    align_corners = False
    align_mode = 0
    data_format = "NWC"
    res = np.array([[[-3.87605909], [-4.72379613], [-6.4192702], [-7.26700723]]])
    obj.run(
        res=res,
        x=x,
        size=size,
        scale_factor=scale_factor,
        mode=mode,
        align_corners=align_corners,
        align_mode=align_mode,
        data_format=data_format,
    )


@pytest.mark.api_nn_functional_upsample_parameters
def test_func_upsample5():
    """
    default
    """
    x = randtool("float", -10, 10, [1, 2, 2, 1])
    size = None
    scale_factor = np.array([2, 1]).astype(np.int32)
    mode = "bicubic"
    align_corners = False
    align_mode = 0
    data_format = "NHWC"
    res = np.array(
        [
            [
                [[1.47893546], [-7.46065862]],
                [[2.8508321], [-5.21906395]],
                [[5.11042658], [-1.52702567]],
                [[6.48232322], [0.714569]],
            ]
        ]
    )
    obj.run(
        res=res,
        x=x,
        size=size,
        scale_factor=scale_factor,
        mode=mode,
        align_corners=align_corners,
        align_mode=align_mode,
        data_format=data_format,
    )


# np.random.seed(33)
# paddle.seed(33)
# x = randtool("float", -10, 10, [1, 2, 2, 1])
# x = paddle.to_tensor(x)
# size = paddle.to_tensor(np.array([4, 4]))
# scale_factor = None
# mode = 'bilinear'
# align_corners = True
# align_mode = 1
# data_format = 'NHWC'
# out = paddle.nn.functional.upsample(x=x,
#              size=size, scale_factor=scale_factor,
#              mode=mode, align_corners=align_corners,
#              align_mode=align_mode, data_format=data_format)
# print(out)
