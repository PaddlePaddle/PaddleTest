#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test initializer KaimingUniform
"""
from apibase import APIBase
from apibase import randtool
import paddle
import pytest
import numpy as np


class TestinitializerKaimingUniform(APIBase):
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


obj = TestinitializerKaimingUniform(paddle.nn.Conv2DTranspose)
obj.places = [paddle.CUDAPlace(0)]
obj.enable_backward = False


@pytest.mark.api_initializer_kaiming_uniform_vartype
def test_initializer_kaiming_uniform_base():
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
    res = np.array(
        [[[[1.1869886, 2.3453705], [1.2281079, 1.0826794]]], [[[0.24804217, 1.8928909], [1.5467637, 1.5817381]]]]
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
            dilation=dilation,
            weight_attr=paddle.nn.initializer.KaimingUniform(),
            bias_attr=False,
        )


@pytest.mark.api_initializer_kaiming_uniform_parameters
def test_initializer_kaiming_uniform1():
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
    res = np.array(
        [
            [
                [
                    [0.36511588, 1.2269412, 1.1467363, 0.49294177],
                    [-0.52867115, 1.1869886, 2.3453705, 0.20336965],
                    [0.27377236, 1.2281079, 1.0826794, 0.16129653],
                    [0.73901653, 1.1044829, 0.68764347, 0.5591744],
                ]
            ],
            [
                [
                    [0.53415686, 1.0935286, 1.0967228, 0.31516552],
                    [-0.577219, 0.24804217, 1.8928909, 0.52769375],
                    [0.25920278, 1.5467637, 1.5817381, -0.00237349],
                    [0.64002484, 1.5542146, -0.01482837, 0.5163526],
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
            dilation=dilation,
            weight_attr=paddle.nn.initializer.KaimingUniform(),
            bias_attr=False,
        )


@pytest.mark.api_initializer_kaiming_uniform_parameters
def test_initializer_kaiming_uniform2():
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
    res = np.array(
        [
            [
                [
                    [0.39888406, 0.14239825, 0.84748095, 1.0942918, 0.7107052, 0.3309278],
                    [-0.9275552, -0.49287915, 0.6899886, 0.5361054, 0.00426798, -0.12755814],
                    [-0.75464916, -0.761385, 0.59295094, 0.64171284, -0.0829928, -0.01879808],
                    [1.0284215, 0.67908627, 0.7174556, 0.04409777, 0.47986156, 0.18009461],
                ]
            ],
            [
                [
                    [0.3629937, -0.66693234, 0.7846964, 0.919392, 0.3969715, 0.6188619],
                    [-0.94021267, -0.453328, 0.5836061, 0.5021896, 0.07433788, -0.09116814],
                    [-0.65587324, -0.4368534, 0.44602627, 0.9121258, 0.00741166, -0.34975865],
                    [0.915076, 0.69684297, 0.84074783, 0.22948253, 0.43271816, 0.34738517],
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
            dilation=dilation,
            weight_attr=paddle.nn.initializer.KaimingUniform(),
            bias_attr=False,
        )


@pytest.mark.api_initializer_kaiming_uniform_parameters
def test_initializer_kaiming_uniform3():
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
                    [-4.38281178e-01, -4.37821895e-01, -7.32276082e-01],
                    [4.86173362e-01, 2.27625296e-01, -2.59767264e-01],
                    [-1.93133652e-01, -1.09906271e-01, -3.99541646e-01],
                    [3.90357763e-01, 2.50917017e-01, -2.10236534e-01],
                    [-1.66459933e-01, -1.45261362e-01, -8.03355426e-02],
                    [0.00000000e00, 0.00000000e00, 0.00000000e00],
                ],
                [
                    [4.16045934e-01, 1.02529573e00, 1.14240563e00],
                    [6.81231797e-01, 9.38934982e-01, 5.39202034e-01],
                    [1.10616410e00, 8.86038125e-01, -3.52791011e-01],
                    [6.99367344e-01, 8.64059448e-01, 5.56699514e-01],
                    [3.66715699e-01, 2.19170019e-01, -3.62740457e-01],
                    [0.00000000e00, 0.00000000e00, 0.00000000e00],
                ],
                [
                    [-6.07524477e-02, -5.88069201e-01, -3.91584873e-01],
                    [3.03896666e-01, 3.38027060e-01, 2.98933475e-03],
                    [-6.38658881e-01, 3.44366312e-01, -2.09073260e-01],
                    [5.23878217e-01, 4.12077278e-01, -2.26563677e-01],
                    [-5.02495617e-02, -4.83799696e-01, -1.10485300e-01],
                    [0.00000000e00, 0.00000000e00, 0.00000000e00],
                ],
                [
                    [3.44050884e-01, 5.26292741e-01, 6.79247379e-01],
                    [-2.31652230e-01, 2.42921382e-01, 9.89836827e-02],
                    [9.40626144e-01, 7.31656134e-01, -2.36744508e-01],
                    [5.12221098e-01, -5.75653791e-01, 2.47303262e-01],
                    [6.00593150e-01, -7.43201613e-01, -2.75598258e-01],
                    [0.00000000e00, 0.00000000e00, 0.00000000e00],
                ],
            ],
            [
                [
                    [-3.05655897e-01, -1.98066980e-02, -3.36162686e-01],
                    [3.56560707e-01, 3.24391574e-01, -3.46581712e-02],
                    [-3.94652635e-01, -3.80285352e-01, -3.03185761e-01],
                    [3.13358665e-01, 1.48906186e-01, 8.57896805e-02],
                    [7.47697894e-03, -6.03874698e-02, 1.92623854e-01],
                    [0.00000000e00, 0.00000000e00, 0.00000000e00],
                ],
                [
                    [-1.79684415e-01, 8.40980768e-01, 8.27742815e-01],
                    [1.49739528e00, 2.24884614e-01, 4.75729644e-01],
                    [1.43841863e00, -1.05765268e-01, 1.47355691e-01],
                    [1.09629488e00, 6.38747394e-01, 5.55811403e-03],
                    [4.43941593e-01, 1.64510027e-01, -6.67716503e-01],
                    [0.00000000e00, 0.00000000e00, 0.00000000e00],
                ],
                [
                    [-5.73447943e-01, -6.35900348e-02, -6.28582120e-01],
                    [1.03080893e00, 1.94387138e-01, -1.30640492e-01],
                    [-7.12517500e-01, 3.60393256e-01, 6.83212857e-05],
                    [3.29666167e-01, 2.75249660e-01, 7.92776346e-02],
                    [2.28361487e-02, -4.34221894e-01, 9.45379362e-02],
                    [0.00000000e00, 0.00000000e00, 0.00000000e00],
                ],
                [
                    [1.38190234e00, 3.81823868e-01, 9.42306817e-01],
                    [1.10175654e-01, -8.40548426e-02, 2.55717456e-01],
                    [1.37264121e00, 4.58975524e-01, -5.76400518e-01],
                    [5.42156637e-01, -5.89082181e-01, -3.88880372e-02],
                    [4.69195813e-01, -6.08345628e-01, -1.66000023e-01],
                    [0.00000000e00, 0.00000000e00, 0.00000000e00],
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
            dilation=dilation,
            groups=groups,
            data_format="NHWC",
            output_padding=output_padding,
            weight_attr=paddle.nn.initializer.KaimingUniform(),
            bias_attr=False,
        )
