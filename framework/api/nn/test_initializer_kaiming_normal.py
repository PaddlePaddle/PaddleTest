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
    obj.static = False
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
                    [[-0.7909662, -1.1310656], [-0.8929211, -1.4893157]],
                    [[-0.34479648, -1.6304922], [-1.3817989, -1.3706139]],
                ]
            ],
            [
                [
                    [[-0.47850963, -0.72494894], [-0.8955704, -0.46525073]],
                    [[-2.0812037, -1.1279415], [-1.2584791, -0.71123725]],
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
    obj.dygraph = False
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
    obj.static = False
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
                        [-1.2293931e00, 1.6474673e-01, -2.8401414e-01],
                        [-6.2907767e-01, 1.2981658e-01, -3.9865896e-01],
                        [1.7846362e-01, 1.2978742e-01, -5.2754354e-01],
                    ],
                    [
                        [-5.9712392e-01, 1.9373848e-01, 3.2503450e-01],
                        [-1.7240536e-01, -1.5997417e-02, -3.7467605e-01],
                        [-5.6236964e-02, -3.8357538e-01, -5.8548778e-01],
                    ],
                    [
                        [-1.6922435e-02, -2.1196042e-01, -4.0865082e-01],
                        [-7.9716820e-01, -8.4401536e-01, -2.0227887e-01],
                        [-4.0591261e-01, -1.1363892e-01, -8.6436117e-01],
                    ],
                ],
                [
                    [
                        [-7.5638346e-02, -1.1987197e00, -6.4783347e-01],
                        [-1.6888726e-01, -1.4597490e00, -8.9046276e-01],
                        [-8.1787658e-01, -1.4053460e00, -1.5710472e00],
                    ],
                    [
                        [-6.7849517e-01, -1.2692145e00, -1.1689229e00],
                        [-1.5266550e00, -1.2151806e00, -1.3315181e-01],
                        [-4.8299670e-01, -6.5576518e-01, -1.1370872e00],
                    ],
                    [
                        [-7.7800590e-01, -8.1772983e-01, -8.8980669e-01],
                        [-7.1756750e-01, -1.1117781e00, -7.7925450e-01],
                        [-8.4614724e-01, -4.3888152e-01, -1.9418462e-01],
                    ],
                ],
                [
                    [
                        [1.8738854e-01, -6.3205600e-02, -1.2471273e00],
                        [-1.0930245e00, -6.3434601e-01, -1.9348118e-01],
                        [-1.8029436e-01, -4.1969761e-01, -2.0330581e-03],
                    ],
                    [
                        [-6.0127503e-01, -8.4321177e-01, 3.0554351e-01],
                        [-2.4791284e-01, -4.1300362e-01, -4.5582914e-01],
                        [-5.5625337e-01, -1.8989690e-01, -8.2144463e-01],
                    ],
                    [
                        [-1.3463125e-01, -6.5465337e-01, -9.0536791e-01],
                        [-4.8383191e-01, -9.1629487e-01, -4.9039146e-01],
                        [-4.3889529e-01, -6.4557642e-01, -7.4583568e-02],
                    ],
                ],
            ],
            [
                [
                    [
                        [-1.0147899e-01, -6.2926716e-01, -2.4559475e-02],
                        [-1.2822889e00, -6.7141855e-01, -1.0656639e-01],
                        [-3.7364462e-01, 1.0084603e00, -1.3760528e-01],
                    ],
                    [
                        [5.8644617e-01, 6.9772981e-02, -2.1342130e-01],
                        [1.1715784e-02, 2.2482160e-01, 1.0146233e-02],
                        [-3.7499946e-01, -3.9224863e-01, -7.4025083e-01],
                    ],
                    [
                        [7.7382021e-02, 8.0774806e-02, -3.2082900e-01],
                        [-2.6685533e-01, -1.5581343e-02, -4.0582401e-01],
                        [-5.1589859e-01, -2.0472760e-01, 1.0251662e-01],
                    ],
                ],
                [
                    [
                        [-8.1672594e-02, -6.4358830e-01, -7.4224633e-01],
                        [-3.9935866e-01, -2.6977694e-01, -1.5822317e-01],
                        [-1.2727587e00, -1.8429492e00, -1.3995315e00],
                    ],
                    [
                        [-1.4347632e00, -7.1152371e-01, -1.5185101e00],
                        [-1.4802868e00, -3.7698421e-01, -4.6810555e-01],
                        [-5.8762294e-01, -1.0453181e00, -1.3732103e00],
                    ],
                    [
                        [-2.9586774e-01, -1.7793399e00, -1.4043761e00],
                        [-2.4744413e00, -1.0216242e00, -3.9117241e-01],
                        [-5.4959458e-01, -1.7778242e-01, -6.2528324e-01],
                    ],
                ],
                [
                    [
                        [-3.9135692e-01, -7.4654871e-01, -8.5980344e-01],
                        [-5.5596483e-01, -8.6022276e-01, -4.7187939e-01],
                        [2.6867282e-01, -7.4003237e-01, -6.0833687e-01],
                    ],
                    [
                        [-3.8614148e-01, -6.1232603e-01, -6.6018879e-01],
                        [-6.9704473e-02, 1.0533802e-01, -4.2883199e-01],
                        [-3.4579116e-01, -3.2097179e-01, -3.6408600e-01],
                    ],
                    [
                        [-7.6128328e-01, -4.4913951e-01, -5.0139654e-01],
                        [-7.4650884e-01, -1.3220979e00, -2.2468255e-01],
                        [-3.5459071e-01, -9.0284270e-01, -1.2002969e00],
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
    obj.dygraph = False
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
    obj.static = False
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4, 4, 4])
    in_channels = 3
    out_channels = 1
    kernel_size = [3, 3, 3]
    stride = 2
    padding = 1
    res = np.array(
        [
            [
                [
                    [[-0.09436481, -0.52511615], [-0.2361718, -1.0775778]],
                    [[-0.6832304, -1.0608681], [-1.0911332, -1.3706139]],
                ]
            ],
            [
                [
                    [[-0.4708487, -0.9790309], [-1.0599958, -0.71658564]],
                    [[-0.67799956, -0.91483], [-1.4096593, -0.71123725]],
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
def test_initializer_kaiming_normal5():
    """
    kernel_size = [3, 3, 3] stride = 2 padding=0 groups=3
    fan_in=None
    """
    obj.dygraph = False
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
    obj.static = False
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
                    [[-0.7909662, -1.1310656], [-0.8929211, -1.4893157]],
                    [[-0.34479648, -1.6304922], [-1.3817989, -1.3706139]],
                ],
                [
                    [[-1.1571394, -0.862601], [-0.35028014, -1.669127]],
                    [[-0.6303307, -0.5712432], [-0.94386023, -0.79027134]],
                ],
            ],
            [
                [
                    [[-0.47850963, -0.72494894], [-0.8955704, -0.46525073]],
                    [[-2.0812037, -1.1279415], [-1.2584791, -0.71123725]],
                ],
                [
                    [[-0.26359233, -0.77682096], [0.08888325, -1.1907777]],
                    [[-0.711984, -0.962748], [-1.0660995, -0.72753334]],
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
def test_initializer_kaiming_normal7():
    """
    kernel_size = [3, 3, 3] stride = 1 padding=1 w=0.7 b=-0.3 data_format="NDHWC"
    fan_in=None
    """
    obj.dygraph = False
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
    obj.static = False
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
                    [[-0.7709425, -1.1404827, -0.668229, -0.52970976], [-1.011386, -1.3369389, -1.4183304, -0.9493112]],
                    [
                        [-1.6819147, -1.4077313, -1.1920983, -1.0478842],
                        [-1.9705657, -1.3817989, -1.3706139, -1.3906765],
                    ],
                ]
            ],
            [
                [
                    [[-1.9058132, -1.3477542, -2.2974548, -1.4315804], [-1.6657716, -0.9627812, -0.5090107, -1.208206]],
                    [
                        [-1.6399553, -1.0677788, -1.1592116, -1.3022541],
                        [-1.9252254, -1.2584791, -0.71123725, -1.4129348],
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
    obj.dygraph = False
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
    obj.static = False
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
                    [
                        [-0.9033673, -0.59911644, -0.9873681, -0.5820413],
                        [-0.72792435, -1.2635299, -1.4326847, -1.1017292],
                    ],
                    [
                        [-1.1342155, -0.85980403, -0.98183185, -1.1548347],
                        [-1.2128533, -1.3817989, -1.3706139, -1.0355732],
                    ],
                ]
            ],
            [
                [
                    [
                        [-0.5132503, -1.251071, -1.0599562, -0.46638155],
                        [-1.8439292, -1.3434633, -0.24912918, -0.79966164],
                    ],
                    [
                        [-1.1400757, -0.7411515, -1.0659547, -1.1601791],
                        [-1.6739165, -1.2584791, -0.71123725, -0.27138665],
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
    obj.dygraph = False
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4, 4, 4])
    in_channels = 3
    out_channels = 1
    kernel_size = [3, 3, 3]
    stride = [2, 2, 1]
    padding = 1
    dilation = 2
    padding_mode = "zeros"
    res = np.array([[[[[0.45051473, 0.24151678]]]], [[[[0.27564842, 0.269262375]]]]])
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
    obj.static = False
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4, 4, 4])
    in_channels = 3
    out_channels = 1
    kernel_size = [3, 3, 3]
    stride = [2, 2, 1]
    padding = 1
    dilation = [2, 2, 2]
    padding_mode = "zeros"
    res = np.array([[[[[-0.12529811, -0.6268066]]]], [[[[-0.27629152, -0.6106175]]]]])
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
    obj.dygraph = False
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
    obj.static = False
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
            [[[[-0.26683137, -0.26825666, -0.706546, -0.6865949], [-1.0431741, -0.745999, 0.11286625, -0.14300884]]]],
            [
                [
                    [
                        [-0.6431576, -0.23594266, -0.9079815, -0.6381485],
                        [-0.72473156, -0.29327083, -0.23808132, -0.2543921],
                    ]
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
    obj.dygraph = False
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
    obj.static = False
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
            [[[[-2.1922429, -2.203953, -5.8048663, -5.640952], [-8.570548, -6.129007, 0.92729086, -1.1749378]]]],
            [[[[-5.284079, -1.9384664, -7.4598293, -5.2429237], [-5.954276, -2.409465, -1.9560382, -2.0900443]]]],
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
    obj.dygraph = False
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
