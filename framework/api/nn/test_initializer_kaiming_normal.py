#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test initializer_kaiming_normal
"""
from apibase import APIBase
from apibase import randtool
import paddle
import pytest
import numpy as np


class TestinitializerKaimingNormal(APIBase):
    """
    test
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32]
        self.delta = 0.005
        self.rtol = 0.005


obj = TestinitializerKaimingNormal(paddle.nn.Conv3D)
obj.places = [paddle.CUDAPlace(0)]
obj.enable_backward = False


@pytest.mark.api_initializer_kaiming_normal_vartype
def test_initializer_kaiming_normal_base():
    """
    base
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4, 4, 4])
    in_channels = 3
    out_channels = 1
    kernel_size = [3, 3, 3]
    stride = 1
    padding = 0
    res = np.array(
        [
            [
                [
                    [[0.6073261, 0.45953372], [-0.14565772, 1.4175302]],
                    [[-0.37328905, 0.14211036], [-0.11755048, 0.99755657]],
                ]
            ],
            [
                [
                    [[0.43749335, 0.84751886], [0.4232624, 0.53650147]],
                    [[0.9127629, 0.37381193], [0.12733835, 0.09655958]],
                ]
            ],
        ]
    )

    if paddle.device.is_compiled_with_cuda() is True:
        obj.base(
            res=res,
            data=x,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            weight_attr=paddle.nn.initializer.KaimingNormal(),
            bias_attr=False,
        )


@pytest.mark.api_initializer_kaiming_normal_parameters
def test_initializer_kaiming_normal1():
    """
    kernel_size = [2, 2, 2]
    fan_in=None
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4, 4, 4])
    in_channels = 3
    out_channels = 1
    kernel_size = [2, 2, 2]
    stride = 1
    padding = 0
    res = np.array(
        [
            [
                [
                    [
                        [1.8079153, 1.3419943, 1.6368787],
                        [1.3206803, 1.3285301, 2.5949469],
                        [1.1816999, 1.5280874, 1.4093121],
                    ],
                    [
                        [0.9836161, 0.4292852, 0.96869326],
                        [1.2596482, 0.89616156, 2.050052],
                        [1.9295127, 1.9959155, 1.4872673],
                    ],
                    [
                        [1.9612684, 1.8047585, 1.1946392],
                        [2.1696332, 1.7041223, 1.3986557],
                        [1.9911445, 1.2479898, 1.019885],
                    ],
                ]
            ],
            [
                [
                    [
                        [1.6421847, 2.0826735, 0.9093389],
                        [1.3807209, 0.97100455, 0.62313694],
                        [1.1052526, -0.12076898, 0.6073525],
                    ],
                    [
                        [1.3408954, 1.6324313, 0.86166024],
                        [1.215758, 0.76346004, 1.9437813],
                        [1.4425654, 2.0354588, 2.2760768],
                    ],
                    [
                        [1.612518, 1.6097023, 1.3900883],
                        [1.2166656, 1.3412042, 1.0103409],
                        [2.384087, 1.6143353, 0.63767433],
                    ],
                ]
            ],
        ]
    )
    if paddle.device.is_compiled_with_cuda() is True:
        obj.run(
            res=res,
            data=x,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            weight_attr=paddle.nn.initializer.KaimingNormal(),
            bias_attr=False,
        )


@pytest.mark.api_initializer_kaiming_normal_parameters
def test_initializer_kaiming_normal2():
    """
    kernel_size = [2, 2, 2], out_channels = 3
    fan_in=None
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4, 4, 4])
    in_channels = 3
    out_channels = 3
    kernel_size = [2, 2, 2]
    stride = 1
    padding = 0
    res = np.array(
        [
            [
                [
                    [
                        [1.8079153, 1.3419943, 1.6368787],
                        [1.3206803, 1.3285301, 2.5949469],
                        [1.1816999, 1.5280874, 1.4093121],
                    ],
                    [
                        [0.9836161, 0.4292852, 0.96869326],
                        [1.2596482, 0.89616156, 2.050052],
                        [1.9295127, 1.9959155, 1.4872673],
                    ],
                    [
                        [1.9612684, 1.8047585, 1.1946392],
                        [2.1696332, 1.7041223, 1.3986557],
                        [1.9911445, 1.2479898, 1.019885],
                    ],
                ],
                [
                    [
                        [0.7471266, 0.45052794, -0.3301444],
                        [-0.2506416, -0.07563769, 0.6927293],
                        [0.4357691, 0.39012286, 0.75422764],
                    ],
                    [
                        [-0.12711209, 0.28128624, 0.78554773],
                        [-0.06531468, 0.07282922, 0.39616638],
                        [0.46107677, 0.6726326, -0.25717333],
                    ],
                    [
                        [0.02403371, 0.32709777, 0.3895478],
                        [0.08395495, -0.5431423, 0.21213472],
                        [0.47807518, 0.10497802, 0.45547783],
                    ],
                ],
                [
                    [
                        [-0.11409266, -0.65414387, -0.02397479],
                        [0.13901153, -0.3198583, -0.01233158],
                        [-0.395255, -0.90410966, -0.75299025],
                    ],
                    [
                        [-0.6351704, -0.29878914, -0.38095522],
                        [-0.672151, 0.17700364, -0.6064391],
                        [-0.96716386, -0.2632199, -0.08827767],
                    ],
                    [
                        [-0.4043926, -0.4363293, -0.634597],
                        [-0.80965, -0.5825709, -0.7071475],
                        [-0.43602312, -0.3779117, -0.38654584],
                    ],
                ],
            ],
            [
                [
                    [
                        [1.6421847, 2.0826735, 0.9093389],
                        [1.3807209, 0.97100455, 0.62313694],
                        [1.1052526, -0.12076898, 0.6073525],
                    ],
                    [
                        [1.3408954, 1.6324313, 0.86166024],
                        [1.215758, 0.76346004, 1.9437813],
                        [1.4425654, 2.0354588, 2.2760768],
                    ],
                    [
                        [1.612518, 1.6097023, 1.3900883],
                        [1.2166656, 1.3412042, 1.0103409],
                        [2.384087, 1.6143353, 0.63767433],
                    ],
                ],
                [
                    [
                        [0.9344932, 0.7502791, 0.9356491],
                        [0.83077097, -0.1439792, 0.91063046],
                        [1.0858068, -0.4682934, -0.3116704],
                    ],
                    [
                        [-0.17357627, 0.22383317, 0.22154495],
                        [0.91086394, 1.3141352, 0.77393717],
                        [-0.22694102, 0.6101852, 0.53859717],
                    ],
                    [
                        [0.14935881, 0.16077325, -0.1769408],
                        [0.16654044, 0.13467762, 0.27446923],
                        [0.41345596, 0.12491152, -0.01616757],
                    ],
                ],
                [
                    [
                        [-0.7843534, -1.361019, -0.7209854],
                        [-0.01214989, -0.091493, -0.2761633],
                        [-1.0982413, -0.17764747, 0.60795546],
                    ],
                    [
                        [0.21671566, -0.0447642, -0.5366475],
                        [-0.4699743, -0.76216847, -0.5679674],
                        [-0.8876607, -0.75843513, -0.5225089],
                    ],
                    [
                        [-0.6694971, -0.55462015, -0.6366767],
                        [-0.58757335, -0.7921887, -1.32232],
                        [-0.3245315, -0.18136731, -0.4579403],
                    ],
                ],
            ],
        ]
    )
    if paddle.device.is_compiled_with_cuda() is True:
        obj.run(
            res=res,
            data=x,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            weight_attr=paddle.nn.initializer.KaimingNormal(),
            bias_attr=False,
        )


@pytest.mark.api_initializer_kaiming_normal_parameters
def test_initializer_kaiming_normal3():
    """
    kernel_size = [3, 3, 3] stride = 2
    fan_in=None
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4, 4, 4])
    in_channels = 3
    out_channels = 1
    kernel_size = [3, 3, 3]
    stride = 2
    padding = 0
    res = np.array([[[[[0.6073261]]]], [[[[0.43749335]]]]])
    if paddle.device.is_compiled_with_cuda() is True:
        obj.run(
            res=res,
            data=x,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            weight_attr=paddle.nn.initializer.KaimingNormal(),
            bias_attr=False,
        )


@pytest.mark.api_initializer_kaiming_normal_parameters
def test_initializer_kaiming_normal4():
    """
    kernel_size = [3, 3, 3] stride = 2 padding=1
    fan_in=None
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4, 4, 4])
    in_channels = 3
    out_channels = 1
    kernel_size = [3, 3, 3]
    stride = 2
    padding = 1
    res = np.array(
        [
            [[[[0.5605063, -0.01152094], [0.594596, 0.2942711]], [[0.04425821, 0.34833288], [0.68170136, 0.99755657]]]],
            [[[[0.1066306, 0.13953194], [0.57458216, 0.2565436]], [[0.09446661, 0.78999925], [0.8357021, 0.09655958]]]],
        ]
    )
    if paddle.device.is_compiled_with_cuda() is True:
        obj.run(
            res=res,
            data=x,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            weight_attr=paddle.nn.initializer.KaimingNormal(),
            bias_attr=False,
        )


@pytest.mark.api_initializer_kaiming_normal_parameters
def test_initializer_kaiming_normal5():
    """
    kernel_size = [3, 3, 3] stride = 2 padding=0 groups=3
    fan_in=None
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4, 4, 4])
    in_channels = 3
    out_channels = 6
    kernel_size = [3, 3, 3]
    stride = 2
    padding = 0
    groups = 3
    res = np.array(
        [
            [[[[1.3194628]]], [[[0.10306186]]], [[[-1.5771013]]], [[[0.58456945]]], [[[0.50026155]]], [[[-1.4732479]]]],
            [[[[1.0636401]]], [[[0.20146653]]], [[[-0.9627709]]], [[[1.5790479]]], [[[0.70708376]]], [[[-1.161625]]]],
        ]
    )
    if paddle.device.is_compiled_with_cuda() is True:
        obj.run(
            res=res,
            data=x,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            weight_attr=paddle.nn.initializer.KaimingNormal(),
            bias_attr=False,
        )


@pytest.mark.api_initializer_kaiming_normal_parameters
def test_initializer_kaiming_normal6():
    """
    kernel_size = [3, 3, 3] stride = 1 padding=1 w=0.7 b=-0.3
    fan_in=None
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4, 4, 4])
    in_channels = 3
    out_channels = 2
    kernel_size = [3, 3, 3]
    stride = 1
    padding = 0
    res = np.array(
        [
            [
                [
                    [[0.6073261, 0.45953372], [-0.14565772, 1.4175302]],
                    [[-0.37328905, 0.14211036], [-0.11755048, 0.99755657]],
                ],
                [[[0.43029046, 1.1630684], [0.31839165, 0.45435104]], [[1.4618211, 1.23204], [1.1701092, 0.7978793]]],
            ],
            [
                [
                    [[0.43749335, 0.84751886], [0.4232624, 0.53650147]],
                    [[0.9127629, 0.37381193], [0.12733835, 0.09655958]],
                ],
                [[[0.5512369, 0.7909282], [0.18140778, 0.47810355]], [[1.1312029, 0.74829596], [0.720579, 0.6829458]]],
            ],
        ]
    )
    if paddle.device.is_compiled_with_cuda() is True:
        obj.run(
            res=res,
            data=x,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            weight_attr=paddle.nn.initializer.KaimingNormal(),
            bias_attr=False,
        )


@pytest.mark.api_initializer_kaiming_normal_parameters
def test_initializer_kaiming_normal7():
    """
    kernel_size = [3, 3, 3] stride = 1 padding=1 w=0.7 b=-0.3 data_format="NDHWC"
    fan_in=None
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4, 4, 4]).transpose(0, 2, 3, 4, 1)
    in_channels = 3
    out_channels = 2
    kernel_size = [3, 3, 3]
    stride = 1
    padding = 0
    data_format = "NDHWC"
    res = np.array(
        [
            [
                [
                    [[0.6073261, 0.43029046], [0.45953372, 1.1630684]],
                    [[-0.14565772, 0.31839165], [1.4175302, 0.45435104]],
                ],
                [
                    [[-0.37328905, 1.4618211], [0.14211036, 1.23204]],
                    [[-0.11755048, 1.1701092], [0.99755657, 0.7978793]],
                ],
            ],
            [
                [
                    [[0.43749335, 0.5512369], [0.84751886, 0.7909282]],
                    [[0.4232624, 0.18140778], [0.53650147, 0.47810355]],
                ],
                [[[0.9127629, 1.1312029], [0.37381193, 0.74829596]], [[0.12733835, 0.720579], [0.09655958, 0.6829458]]],
            ],
        ]
    )
    if paddle.device.is_compiled_with_cuda() is True:
        obj.run(
            res=res,
            data=x,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            data_format=data_format,
            weight_attr=paddle.nn.initializer.KaimingNormal(),
            bias_attr=False,
        )


@pytest.mark.api_initializer_kaiming_normal_parameters
def test_initializer_kaiming_normal8():
    """
    padding_mode = "reflect"
    fan_in=None
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4, 4, 4])
    in_channels = 3
    out_channels = 1
    kernel_size = [3, 3, 3]
    stride = [2, 2, 1]
    padding = 1
    padding_mode = "reflect"
    res = np.array(
        [
            [
                [
                    [[1.1522751, 0.64449316, -0.1596964, 0.3406393], [0.85893476, -0.3124804, 0.82826024, 0.5503867]],
                    [[0.66155595, 0.3669618, 0.31739455, 0.08125566], [1.0070366, -0.11755048, 0.99755657, 0.4241258]],
                ]
            ],
            [
                [
                    [[0.78231233, -0.03063385, 0.6058931, 0.2871665], [0.3692535, 0.17527363, 0.14924596, 0.85669655]],
                    [[0.4449456, -0.6211853, 0.58943945, 0.33447614], [0.19076191, 0.12733835, 0.09655958, 0.7136871]],
                ]
            ],
        ]
    )
    if paddle.device.is_compiled_with_cuda() is True:
        obj.run(
            res=res,
            data=x,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            padding_mode=padding_mode,
            weight_attr=paddle.nn.initializer.KaimingNormal(),
            bias_attr=False,
        )


@pytest.mark.api_initializer_kaiming_normal_parameters
def test_initializer_kaiming_normal9():
    """
    padding_mode = "replicate"
    fan_in=None
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4, 4, 4])
    in_channels = 3
    out_channels = 1
    kernel_size = [3, 3, 3]
    stride = [2, 2, 1]
    padding = 1
    padding_mode = "replicate"
    res = np.array(
        [
            [
                [
                    [[1.0748712, 1.2466089, 0.2374663, 0.21809292], [0.26393467, -0.24257615, 1.0476159, 0.4572817]],
                    [[0.6566678, 0.35547423, 0.20397957, 0.11591544], [0.5150983, -0.11755048, 0.99755657, 0.41333783]],
                ]
            ],
            [
                [
                    [[0.7300893, 0.9885114, 0.63725376, 0.8691806], [0.25921577, -0.41466352, 0.28727567, 1.3703382]],
                    [[-0.01092687, -0.3061905, 1.0700703, 0.25076237], [0.7927509, 0.12733835, 0.09655958, 1.0127653]],
                ]
            ],
        ]
    )
    if paddle.device.is_compiled_with_cuda() is True:
        obj.run(
            res=res,
            data=x,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            padding_mode=padding_mode,
            weight_attr=paddle.nn.initializer.KaimingNormal(),
            bias_attr=False,
        )


@pytest.mark.api_initializer_kaiming_normal_parameters
def test_initializer_kaiming_normal10():
    """
    padding_mode = "circular"
    fan_in=None
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4, 4, 4])
    in_channels = 3
    out_channels = 1
    kernel_size = [3, 3, 3]
    stride = [2, 2, 1]
    padding = 1
    padding_mode = "circular"
    res = np.array(
        [
            [
                [
                    [[1.1493586, 0.2857985, 0.4673859, 0.79607165], [0.92873603, 0.20516342, 0.7362683, 1.0331961]],
                    [[0.3038256, 1.1247545, 0.6776748, 0.12748581], [1.1636142, -0.11755048, 0.99755657, 0.32947105]],
                ]
            ],
            [
                [
                    [[0.26404598, -0.6576977, 0.13014497, 1.3358058], [0.40981585, 0.4426188, 0.16423123, 0.59834445]],
                    [[0.306283, -0.4172607, 0.73213583, 0.6948482], [1.0667173, 0.12733835, 0.09655958, 1.3202523]],
                ]
            ],
        ]
    )
    if paddle.device.is_compiled_with_cuda() is True:
        obj.run(
            res=res,
            data=x,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            padding_mode=padding_mode,
            weight_attr=paddle.nn.initializer.KaimingNormal(),
            bias_attr=False,
        )


@pytest.mark.api_initializer_kaiming_normal_parameters
def test_initializer_kaiming_normal11():
    """
    padding_mode = "zeros" dilation = 2
    fan_in=None
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4, 4, 4])
    in_channels = 3
    out_channels = 1
    kernel_size = [3, 3, 3]
    stride = [2, 2, 1]
    padding = 1
    dilation = 2
    padding_mode = "zeros"
    res = np.array([[[[[0.45051473, 0.24151678]]]], [[[[0.27564842, 0.26926237]]]]])
    if paddle.device.is_compiled_with_cuda() is True:
        obj.run(
            res=res,
            data=x,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            padding_mode=padding_mode,
            dilation=dilation,
            weight_attr=paddle.nn.initializer.KaimingNormal(),
            bias_attr=False,
        )


@pytest.mark.api_initializer_kaiming_normal_parameters
def test_initializer_kaiming_normal12():
    """
    padding_mode = "zeros" dilation = [2, 2, 2]
    fan_in=None
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4, 4, 4])
    in_channels = 3
    out_channels = 1
    kernel_size = [3, 3, 3]
    stride = [2, 2, 1]
    padding = 1
    dilation = [2, 2, 2]
    padding_mode = "zeros"
    res = np.array([[[[[0.45051473, 0.24151678]]]], [[[[0.27564842, 0.26926237]]]]])
    if paddle.device.is_compiled_with_cuda() is True:
        obj.run(
            res=res,
            data=x,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            padding_mode=padding_mode,
            dilation=dilation,
            weight_attr=paddle.nn.initializer.KaimingNormal(),
            bias_attr=False,
        )


@pytest.mark.api_initializer_kaiming_normal_parameters
def test_initializer_kaiming_normal13():
    """
    padding_mode = "zeros" dilation = (2, 2)
    fan_in=None
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4, 4, 4])
    in_channels = 3
    out_channels = 1
    kernel_size = [3, 3, 3]
    stride = [2, 2, 1]
    padding = 1
    dilation = (2, 2, 2)
    padding_mode = "zeros"
    res = np.array([[[[[0.45051473, 0.24151678]]]], [[[[0.27564842, 0.26926237]]]]])
    if paddle.device.is_compiled_with_cuda() is True:
        obj.run(
            res=res,
            data=x,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            padding_mode=padding_mode,
            dilation=dilation,
            weight_attr=paddle.nn.initializer.KaimingNormal(),
            bias_attr=False,
        )


@pytest.mark.api_initializer_kaiming_normal_parameters
def test_initializer_kaiming_normal14():
    """
    padding_mode = "zeros" dilation = (2, 2, 2)  padding = (1, 2, 2)
    fan_in=None
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4, 4, 4])
    in_channels = 3
    out_channels = 1
    kernel_size = [3, 3, 3]
    stride = [2, 2, 1]
    padding = (1, 2, 2)
    dilation = (2, 2, 2)
    padding_mode = "zeros"
    res = np.array(
        [
            [[[[0.27735454, 0.176687, 0.2575486, 0.2179849], [0.4793579, 0.04378546, -0.26692072, 0.01292928]]]],
            [[[[0.43240556, -0.07143027, 0.12640606, 0.29070508], [0.4951439, 0.5538594, -0.0941367, 0.19308226]]]],
        ]
    )
    if paddle.device.is_compiled_with_cuda() is True:
        obj.run(
            res=res,
            data=x,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            padding_mode=padding_mode,
            dilation=dilation,
            weight_attr=paddle.nn.initializer.KaimingNormal(),
            bias_attr=False,
        )


@pytest.mark.api_initializer_kaiming_normal_parameters
def test_initializer_kaiming_normal15():
    """
    padding_mode = "zeros" dilation = (2, 2)  padding = [1, 2]
    fan_in=None
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4, 4, 4])
    in_channels = 3
    out_channels = 1
    kernel_size = [3, 3, 3]
    stride = [2, 2, 1]
    padding = [1, 2, 2]
    dilation = (2, 2, 2)
    padding_mode = "zeros"
    res = np.array(
        [
            [[[[0.27735454, 0.176687, 0.2575486, 0.2179849], [0.4793579, 0.04378546, -0.26692072, 0.01292928]]]],
            [[[[0.43240556, -0.07143027, 0.12640606, 0.29070508], [0.4951439, 0.5538594, -0.0941367, 0.19308226]]]],
        ]
    )
    if paddle.device.is_compiled_with_cuda() is True:
        obj.run(
            res=res,
            data=x,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            padding_mode=padding_mode,
            dilation=dilation,
            weight_attr=paddle.nn.initializer.KaimingNormal(),
            bias_attr=False,
        )


@pytest.mark.api_initializer_kaiming_normal_parameters
def test_initializer_kaiming_normal16():
    """
    padding_mode = "zeros" dilation = (2, 2)  padding = [1, 2]
    fan_in=1.2
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4, 4, 4])
    in_channels = 3
    out_channels = 1
    kernel_size = [3, 3, 3]
    stride = [2, 2, 1]
    padding = [1, 2, 2]
    dilation = (2, 2, 2)
    padding_mode = "zeros"
    res = np.array(
        [
            [[[[2.2786992, 1.4516317, 2.115977, 1.7909288], [3.9383252, 0.3597339, -2.192978, 0.10622477]]]],
            [[[[3.552574, -0.58685935, 1.0385318, 2.3883858], [4.068023, 4.550418, -0.77341205, 1.5863327]]]],
        ]
    )
    if paddle.device.is_compiled_with_cuda() is True:
        obj.run(
            res=res,
            data=x,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            padding_mode=padding_mode,
            dilation=dilation,
            weight_attr=paddle.nn.initializer.KaimingNormal(fan_in=1.2),
            bias_attr=False,
        )


@pytest.mark.api_initializer_kaiming_normal_parameters
def test_initializer_kaiming_normal17():
    """
    padding_mode = "zeros" dilation = (2, 2)  padding = [1, 2]
    fan_in=2.3
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4, 4, 4])
    in_channels = 3
    out_channels = 1
    kernel_size = [3, 3, 3]
    stride = [2, 2, 1]
    padding = [1, 2, 2]
    dilation = (2, 2, 2)
    padding_mode = "zeros"
    res = np.array(
        [
            [[[[1.6459391, 1.0485356, 1.5284024, 1.293615], [2.8447134, 0.25984153, -1.5840213, 0.07672782]]]],
            [[[[2.566078, -0.42389768, 0.7501471, 1.7251672], [2.938395, 3.2868364, -0.55864704, 1.1458318]]]],
        ]
    )
    if paddle.device.is_compiled_with_cuda() is True:
        obj.run(
            res=res,
            data=x,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            padding_mode=padding_mode,
            dilation=dilation,
            weight_attr=paddle.nn.initializer.KaimingNormal(fan_in=2.3),
            bias_attr=False,
        )
