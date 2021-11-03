#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_nn_utils_spectral_norm
"""
from apibase import compare
from apibase import randtool
import paddle
import pytest
import numpy as np

paddle.seed(33)
obj_base = paddle.nn.utils.spectral_norm

obj1 = paddle.nn.Linear(3, 6)
weight1 = np.array(
    [
        [0.72889733, 0.81235313, 0.12548596, 0.64302790, -0.26532084, -0.15424937],
        [0.71701384, -0.06694162, -0.14641583, 0.48437297, -0.44942823, 0.75306857],
        [0.59001660, -0.14815784, 0.16762769, -0.37657404, 0.70948184, -0.18634099],
    ]
).astype(np.float32)
obj1._parameters["weight"].set_value(weight1)

obj2 = paddle.nn.Conv2D(in_channels=3, out_channels=4, kernel_size=[3, 3], stride=1, padding=0)
weight2 = np.array(
    [
        [
            [
                [-0.55039716, 0.28371421, 0.13347910],
                [0.00098233, 0.43559369, -0.14512429],
                [0.21122162, 0.46407086, -0.53618413],
            ],
            [
                [-0.30149207, 0.07910406, 0.15358460],
                [-0.40321645, 0.14424308, -0.13095412],
                [0.06747758, 0.01251858, -0.45162091],
            ],
            [
                [-0.08531693, 0.04468197, 0.30074450],
                [-0.39757499, -0.22381847, -0.30894721],
                [0.03670330, -0.27533460, -0.09633731],
            ],
        ],
        [
            [
                [0.04307014, -0.31103051, 0.02288885],
                [0.11133088, -0.20271860, -0.13357930],
                [0.22855571, -0.16834489, 0.07812120],
            ],
            [
                [-0.13061447, 0.17873897, -0.08831661],
                [0.01345632, 0.29348874, 0.10168455],
                [0.06622937, -0.08446530, -0.30304945],
            ],
            [
                [0.49557865, -0.09265980, -0.17045820],
                [-0.45063367, 0.03004680, -0.28458628],
                [-0.18199448, 0.22933708, -0.41798902],
            ],
        ],
        [
            [
                [0.33591449, -0.17593814, 0.26064998],
                [0.41464412, 0.20778862, 0.08883220],
                [0.24975073, -0.29537794, 0.60702217],
            ],
            [
                [0.11234486, -0.08978345, 0.54455924],
                [0.27078903, -0.19312058, 0.05361092],
                [0.50540131, 0.13008922, -0.18097508],
            ],
            [
                [0.57940012, 0.08052463, -0.07993308],
                [0.00621510, 0.42073262, 0.07991114],
                [0.26453930, 0.09337366, 0.09302550],
            ],
        ],
        [
            [
                [-0.31293857, 0.21733348, -0.03819288],
                [0.08018019, -0.01144855, 0.09929247],
                [-0.05727239, -0.47983477, -0.61332005],
            ],
            [
                [0.03894517, -0.07997172, -0.00454860],
                [0.20150150, -0.88335091, 0.00483579],
                [0.13447969, -0.41225162, 0.07614432],
            ],
            [
                [0.23930417, 0.18523265, 0.33545452],
                [0.22861424, -0.34291700, -0.17685926],
                [-0.42208901, 0.35519785, 0.52931118],
            ],
        ],
    ]
).astype(np.float32)
obj2._parameters["weight"].set_value(weight2)

obj3 = paddle.nn.Conv1D(in_channels=3, out_channels=4, kernel_size=3, stride=1, padding=0)
weight3 = np.array(
    [
        [
            [0.83978403, -0.15663569, 0.39995280],
            [-0.99171132, -0.17835638, 0.01188756],
            [-0.32428807, 0.07860076, 0.00781965],
        ],
        [
            [-0.28639862, 0.20968308, 0.09383135],
            [1.17916155, -0.20114765, -0.13593331],
            [-1.08055127, 0.45732364, 0.33627468],
        ],
        [
            [-0.18641895, -0.09555656, -0.17164250],
            [0.63788563, 0.19814725, 0.12208853],
            [0.14774144, 0.43917039, 0.10739637],
        ],
        [
            [-0.37824255, -0.31575939, -0.48575446],
            [-0.18304695, -0.35072994, -0.12200834],
            [-0.39261225, 0.25912943, 0.60038394],
        ],
    ]
).astype(np.float32)
obj3._parameters["weight"].set_value(weight3)


@pytest.mark.api_nn_SpectralNorm_parameters
def test_utils_spectral_norm_linear():
    """
    utils_spectral_norm with paddle.nn.linear
    """
    # paddle.seed(33)
    np.random.seed(33)
    # inputs = paddle.rand((4, 3))
    inputs = randtool("float", -1, 1, [4, 3]).astype(np.float32)
    spectral_norm = obj_base(obj1, dim=0, n_power_iterations=2)
    weight_u = np.array([0.01091708, 0.73375094, -0.67933077]).astype(np.float32)
    weight_v = np.array([0.18429381, -0.11625154, 0.08476070, -0.38605255, 0.89188033, 0.02914998]).astype(np.float32)
    spectral_norm._buffers["weight_u"].set_value(weight_u)
    spectral_norm._buffers["weight_v"].set_value(weight_v)
    spectral_norm_out = spectral_norm(paddle.to_tensor(inputs))
    exp = np.array(
        [
            [-0.36249605, -0.2504755, -0.05224538, -0.20332088, 0.03471488, 0.02363358],
            [-0.12669484, -0.23059455, -0.18290804, 0.19194576, -0.4353384, 0.49973285],
            [0.10851127, -0.59669757, -0.12859017, -0.20982172, 0.06905889, 0.5093373],
            [0.34786582, -0.03499362, -0.11691605, 0.34253922, -0.37496197, 0.49644187],
        ]
    ).astype(np.float32)
    res = spectral_norm_out.numpy()
    compare(res, exp)


@pytest.mark.api_nn_SpectralNorm_parameters
def test_utils_spectral_norm_Conv2D():
    """
    utils_spectral_norm with paddle.nn.Conv2D
    """
    # paddle.seed(33)
    np.random.seed(33)
    inputs = randtool("float", -1, 1, [1, 3, 4, 4]).astype(np.float32)
    # Conv2D = obj2(in_channels=3, out_channels=4, kernel_size=[3, 3], stride=1, padding=0)
    spectral_norm = obj_base(obj2, dim=1, n_power_iterations=2)
    weight_u = np.array([0.18215272, 0.80801064, 0.56030279]).astype(np.float32)
    weight_v = np.array(
        [
            -0.07316514,
            0.09928722,
            -0.00992621,
            0.32455149,
            0.01056808,
            -0.12354111,
            -0.04251307,
            -0.10059334,
            -0.24861489,
            -0.09038673,
            -0.40843984,
            0.25231028,
            0.13895534,
            0.16004264,
            0.13879651,
            -0.02009732,
            -0.00903338,
            -0.04133186,
            -0.21715638,
            0.08604021,
            0.06142099,
            -0.10293791,
            -0.32196543,
            -0.02826083,
            0.00025799,
            -0.20911011,
            -0.24329987,
            0.30002689,
            0.06330941,
            -0.27194566,
            -0.04207566,
            0.05842301,
            -0.10486081,
            -0.10844701,
            -0.10801455,
            -0.03627804,
        ]
    ).astype(np.float32)
    spectral_norm._buffers["weight_u"].set_value(weight_u)
    spectral_norm._buffers["weight_v"].set_value(weight_v)
    spectral_norm_out = spectral_norm(paddle.to_tensor(inputs))
    exp = np.array(
        [
            [
                [[-0.63234615, 0.24694696], [-0.8783692, 0.32894957]],
                [[0.00753528, 0.32653555], [-0.34002084, 0.32660863]],
                [[-0.08457994, -0.91970307], [0.04903618, 0.07056515]],
                [[0.4485203, -0.9982232], [0.6400923, -1.041528]],
            ]
        ]
    ).astype(np.float32)
    res = spectral_norm_out.numpy()
    compare(res, exp)


@pytest.mark.api_nn_SpectralNorm_parameters
def test_utils_spectral_norm_Conv1D():
    """
    utils_spectral_norm with paddle.nn.Conv1D
    """
    # paddle.seed(33)
    np.random.seed(33)
    inputs = randtool("float", -1, 1, [1, 3, 5]).astype(np.float32)
    # Conv1D = obj3(in_channels=3, out_channels=4, kernel_size=3, stride=1, padding=0)
    spectral_norm = obj_base(obj3, dim=1, n_power_iterations=2)
    weight_u = np.array([0.45380810, -0.27697065, -0.84696251]).astype(np.float32)
    weight_v = np.array(
        [
            -0.08141783,
            0.31601432,
            0.17962831,
            -0.35668275,
            0.25347021,
            0.08989146,
            -0.19923738,
            0.32060334,
            -0.00806985,
            -0.33321515,
            0.17902276,
            -0.61323804,
        ]
    ).astype(np.float32)
    spectral_norm._buffers["weight_u"].set_value(weight_u)
    spectral_norm._buffers["weight_v"].set_value(weight_v)
    spectral_norm_out = spectral_norm(paddle.to_tensor(inputs))
    exp = np.array(
        [
            [
                [0.00261505, 0.3659563, -0.33233038],
                [-1.3028698, -1.1161773, 1.0457263],
                [-0.2631139, -0.4698858, 0.18114574],
                [-0.18374847, -0.11467814, -0.38792706],
            ]
        ]
    ).astype(np.float32)
    res = spectral_norm_out.numpy()
    compare(res, exp)
