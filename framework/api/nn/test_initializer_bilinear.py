#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test paddle.nn.initializer.Bilinear
"""
from apibase import APIBase
from apibase import randtool
import paddle
import pytest
import numpy as np


class TestinitializerBilinear(APIBase):
    """
    test
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32]
        self.delta = 1e-3 * 5
        self.rtol = 1e-3


obj = TestinitializerBilinear(paddle.nn.Conv2DTranspose)


@pytest.mark.api_initializer_bilinear_vartype
def test_initializer_bilinear_base():
    """
    base
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 2, 2]).astype("float32")
    in_channels = 3
    out_channels = 1
    kernel_size = [3, 3]
    stride = 1
    padding = 1
    dilation = 1
    # padding_mode = "zeros"
    # groups = 1
    res = np.array(
        [[[[1.9839268, 3.1209197], [2.2583175, 3.1172738]]], [[[1.8303242, 2.9195924], [2.052437, 2.9040678]]]]
    )
    obj.base(
        res=res,
        data=x,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        weight_attr=paddle.nn.initializer.Bilinear(),
        bias_attr=False,
    )


@pytest.mark.api_initializer_bilinear_parameters
def test_initializer_bilinear1():
    """
    default
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 2, 2]).astype("float32")
    in_channels = 3
    out_channels = 1
    kernel_size = [3, 3]
    stride = 1
    padding = 0
    dilation = 1
    # padding_mode = "zeros"
    # groups = 1
    res = np.array(
        [
            [
                [
                    [0.11245979, 0.6323992, 1.1377196, 0.49070153],
                    [0.42460617, 1.9839268, 3.1209197, 1.4741431],
                    [0.5990598, 2.2583175, 3.1172738, 1.4552795],
                    [0.26168045, 0.9118909, 1.1391748, 0.502172],
                ]
            ],
            [
                [
                    [0.10506642, 0.59365946, 1.0771022, 0.47829214],
                    [0.38941014, 1.8303242, 2.9195924, 1.4414989],
                    [0.53783196, 2.052437, 2.9040678, 1.4257609],
                    [0.2226327, 0.8151463, 1.0609514, 0.49278346],
                ]
            ],
        ]
    )
    obj.run(
        res=res,
        data=x,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        weight_attr=paddle.nn.initializer.Bilinear(),
        bias_attr=False,
    )


@pytest.mark.api_initializer_bilinear_parameters
def test_initializer_bilinear2():
    """
    padding = [1, 0]
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 2, 2]).astype("float32")
    in_channels = 3
    out_channels = 1
    kernel_size = [3, 3]
    stride = 1
    padding = [1, 0]
    dilation = 1
    # groups = 1
    res = np.array(
        [
            [[[0.42460617, 1.9839268, 3.1209197, 1.4741431], [0.5990598, 2.2583175, 3.1172738, 1.4552795]]],
            [[[0.38941014, 1.8303242, 2.9195924, 1.4414989], [0.53783196, 2.052437, 2.9040678, 1.4257609]]],
        ]
    )
    obj.run(
        res=res,
        data=x,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        weight_attr=paddle.nn.initializer.Bilinear(),
        bias_attr=False,
    )


@pytest.mark.api_initializer_bilinear_parameters
def test_initializer_bilinear3():
    """
    dilation = 2
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 2, 2]).astype("float32")
    in_channels = 3
    out_channels = 1
    kernel_size = [3, 3]
    stride = 1
    padding = [1, 0]
    dilation = 2
    # groups = 1
    res = np.array(
        [
            [
                [
                    [0.08722682, 0.10043441, 0.43613407, 0.502172, 0.6105877, 0.7030408],
                    [0.33737937, 0.21030065, 1.2370577, 0.77110237, 1.2370577, 0.77110237],
                    [0.26168045, 0.3013032, 0.95949495, 1.1047784, 0.95949495, 1.1047784],
                    [0.33737937, 0.21030065, 0.7872185, 0.49070153, 0.56229895, 0.3505011],
                ]
            ],
            [
                [
                    [0.0742109, 0.09855669, 0.3710545, 0.49278346, 0.5194763, 0.6898969],
                    [0.31519926, 0.20498234, 1.1557306, 0.751602, 1.1557306, 0.751602],
                    [0.2226327, 0.29567006, 0.8163199, 1.0841236, 0.8163199, 1.0841236],
                    [0.31519926, 0.20498234, 0.7354649, 0.47829214, 0.52533203, 0.34163725],
                ]
            ],
        ]
    )
    obj.run(
        res=res,
        data=x,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        weight_attr=paddle.nn.initializer.Bilinear(),
        bias_attr=False,
    )


@pytest.mark.api_initializer_bilinear_parameters
def test_initializer_bilinear4():
    """
    out_channels = 3 groups=3
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 6, 2, 2]).astype("float32")
    in_channels = 6
    out_channels = 3
    kernel_size = [3, 3]
    stride = 1
    padding = [1, 0]
    dilation = 1
    groups = 3
    res = np.array(
        [
            [
                [[0.23670748, 1.0987233, 1.7734442, 0.9675019], [0.29053277, 1.1321666, 1.757833, 1.0327591]],
                [[0.21304294, 1.1183721, 2.022594, 1.1783693], [0.35412568, 1.523689, 2.3264697, 1.2082624]],
                [[0.36426592, 1.5971556, 2.244474, 0.76977074], [0.4922333, 1.6548989, 1.9370389, 0.6400189]],
            ],
            [
                [[0.27036875, 1.3572879, 2.2191925, 0.93672013], [0.510839, 1.9881839, 2.5278025, 0.8544663]],
                [[0.24372602, 1.0984795, 1.6581774, 0.7442159], [0.3248785, 1.1860256, 1.6347295, 0.7876489]],
                [[0.2492792, 1.2557914, 2.045677, 0.8566852], [0.43473136, 1.6375296, 1.9377347, 0.52277696]],
            ],
        ]
    )
    obj.run(
        res=res,
        data=x,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        weight_attr=paddle.nn.initializer.Bilinear(),
        bias_attr=False,
    )


@pytest.mark.api_initializer_bilinear_parameters
def test_initializer_bilinear5():
    """
    out_channels = 3 groups=3 data_format="NHWC"
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 6, 2, 2]).astype("float32").transpose(0, 2, 3, 1)
    in_channels = 6
    out_channels = 3
    kernel_size = [3, 3]
    stride = 1
    padding = [1, 0]
    dilation = 1
    groups = 3
    res = np.array(
        [
            [
                [
                    [0.23670748, 0.21304294, 0.36426592],
                    [1.0987233, 1.1183721, 1.5971556],
                    [1.7734442, 2.022594, 2.244474],
                    [0.9675019, 1.1783693, 0.76977074],
                ],
                [
                    [0.29053277, 0.35412568, 0.4922333],
                    [1.1321666, 1.523689, 1.6548989],
                    [1.757833, 2.3264697, 1.9370389],
                    [1.0327591, 1.2082624, 0.6400189],
                ],
            ],
            [
                [
                    [0.27036875, 0.24372602, 0.2492792],
                    [1.3572879, 1.0984795, 1.2557914],
                    [2.2191925, 1.6581774, 2.045677],
                    [0.93672013, 0.7442159, 0.8566852],
                ],
                [
                    [0.510839, 0.3248785, 0.43473136],
                    [1.9881839, 1.1860256, 1.6375296],
                    [2.5278025, 1.6347295, 1.9377347],
                    [0.8544663, 0.7876489, 0.52277696],
                ],
            ],
        ]
    )
    obj.run(
        res=res,
        data=x,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        data_format="NHWC",
        weight_attr=paddle.nn.initializer.Bilinear(),
        bias_attr=False,
    )


@pytest.mark.api_initializer_bilinear_parameters
def test_initializer_bilinear6():
    """
    out_channels = 3 groups=3 data_format="NHWC" output_padding=1
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 6, 2, 2]).astype("float32").transpose(0, 2, 3, 1)
    in_channels = 6
    out_channels = 3
    kernel_size = [3, 3]
    stride = 2
    padding = [1, 0]
    dilation = 1
    groups = 3
    output_padding = 1
    res = np.array(
        [
            [
                [
                    [0.20979483, 0.14250156, 0.3002822],
                    [0.7692477, 0.52250576, 1.1010348],
                    [0.88831306, 0.6796299, 1.2401283],
                    [0.43657306, 0.5761219, 0.51000935],
                    [0.43657306, 0.5761219, 0.51000935],
                    [0.0, 0.0, 0.0],
                ],
                [
                    [0.23670748, 0.21304294, 0.36426592],
                    [0.6240845, 0.68521047, 1.020577],
                    [0.7329589, 0.9744518, 1.1245584],
                    [0.6570541, 0.7967998, 0.51009524],
                    [0.72937113, 0.864121, 0.4915838],
                    [0.0, 0.0, 0.0],
                ],
                [
                    [0.08073792, 0.21162412, 0.1919511],
                    [0.29603902, 0.7759551, 0.7038207],
                    [0.52357996, 1.0340612, 0.81514704],
                    [0.8343168, 0.9463888, 0.40819645],
                    [0.8343168, 0.9463888, 0.40819645],
                    [0.0, 0.0, 0.0],
                ],
                [
                    [0.08073792, 0.21162412, 0.1919511],
                    [0.18838847, 0.4937896, 0.4478859],
                    [0.36210412, 0.6108129, 0.43124482],
                    [0.53092885, 0.6022474, 0.25976136],
                    [0.3792349, 0.43017673, 0.18554384],
                    [0.0, 0.0, 0.0],
                ],
            ],
            [
                [
                    [0.15013364, 0.20314977, 0.1565531],
                    [0.55049, 0.74488246, 0.5740281],
                    [0.7012804, 0.8381208, 0.7751013],
                    [0.55289805, 0.34187394, 0.7372685],
                    [0.55289805, 0.34187394, 0.7372685],
                    [0.0, 0.0, 0.0],
                ],
                [
                    [0.27036875, 0.24372602, 0.2492792],
                    [0.95148754, 0.6768974, 0.8289211],
                    [1.2974908, 0.7733325, 1.1281374],
                    [0.6260029, 0.50494325, 0.5544686],
                    [0.6351394, 0.5577392, 0.4545388],
                    [0.0, 0.0, 0.0],
                ],
                [
                    [0.36070538, 0.12172876, 0.27817827],
                    [1.3225864, 0.44633877, 1.0199871],
                    [1.4870816, 0.6187711, 1.0711657],
                    [0.60314906, 0.6322517, 0.18765488],
                    [0.60314906, 0.6322517, 0.18765488],
                    [0.0, 0.0, 0.0],
                ],
                [
                    [0.36070538, 0.12172876, 0.27817827],
                    [0.8416459, 0.28403378, 0.64908266],
                    [0.7656709, 0.37531352, 0.51480913],
                    [0.3838221, 0.40234196, 0.11941674],
                    [0.27415866, 0.28738713, 0.08529767],
                    [0.0, 0.0, 0.0],
                ],
            ],
        ]
    )
    obj.run(
        res=res,
        data=x,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        data_format="NHWC",
        output_padding=output_padding,
        weight_attr=paddle.nn.initializer.Bilinear(),
        bias_attr=False,
    )
