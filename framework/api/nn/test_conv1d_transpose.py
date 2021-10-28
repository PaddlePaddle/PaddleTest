#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test Conv1DTranspose
"""
import platform
from apibase import APIBase
from apibase import randtool
import paddle
import paddle.fluid as fluid
import pytest
import numpy as np


class TestConv1dTranspose(APIBase):
    """
    test
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32]
        self.delta = 1e-3
        # self.debug = True
        # self.static = True
        # enable check grad
        # self.enable_backward = True


obj = TestConv1dTranspose(paddle.nn.Conv1DTranspose)


@pytest.mark.api_nn_Conv1D_transpose_vartype
def test_conv1d_transpose_base():
    """
    base
    """

    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [1, 2, 4])
    in_channels = 2
    out_channels = 1
    kernel_size = [2]
    stride = 1
    padding = 0
    res = np.array([[[1.1189058, 1.7539212, 1.0656176, 1.644154, 1.2135518]]])
    obj.base(
        res=res,
        data=x,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        weight_attr=fluid.initializer.ConstantInitializer(value=1),
        bias_attr=fluid.initializer.ConstantInitializer(value=0),
    )


@pytest.mark.api_nn_Conv1D_parameters
def test_conv1d_transpose():
    """
    default
    """

    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 2])
    in_channels = 3
    out_channels = 2
    kernel_size = [3]
    groups = 1
    stride = 1
    padding = 0
    output_padding = 0
    dilation = 1
    # weight_attr = np.full(shape=[in_channels, out_channels // groups] + kernel_size, fill_value=1)
    # bias_attr = np.full(shape=[out_channels], fill_value=0)
    res = np.array(
        [
            [[1.5298, 2.4252, 2.4252, 0.8953], [1.5298, 2.4252, 2.4252, 0.8953]],
            [[1.6651, 3.4984, 3.4984, 1.8332], [1.6651, 3.4984, 3.4984, 1.8332]],
        ]
    )
    obj.run(
        res=res,
        data=x,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size = kernel_size,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        groups=groups,
        dilation=dilation,
        weight_attr=fluid.initializer.ConstantInitializer(value=1),
        bias_attr=fluid.initializer.ConstantInitializer(value=0),
        data_format="NCL",
    )


@pytest.mark.api_nn_Conv1D_Transpose_parameters
def test_conv1d_transpose1():
    """
    kernel_size = 2
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 2])
    in_channels = 3
    out_channels = 2
    kernel_size = 2
    groups = 1
    stride = 1
    padding = 0
    output_padding = 0
    dilation = 1
    res = np.array(
        [
            [[1.5298, 2.4251, 0.895],[1.5298, 2.4251, 0.895]],
            [[1.6651, 3.4983, 1.833],[1.6651, 3.4984, 1.834]],
        ]
    )

    obj.run(
        res=res,
        data=x,
        in_channels=in_channels,
        out_channels=out_channels,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        kernel_size = kernel_size,
        groups=groups,
        dilation=dilation,
        weight_attr=fluid.initializer.ConstantInitializer(value=1),
        bias_attr=fluid.initializer.ConstantInitializer(value=0),
        data_format="NCL",
    )

@pytest.mark.api_nn_Conv1D_Transpose_parameters
def test_conv1d_transpose2():
    """
    kernel_size = (2) padding = [0]
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 2])
    in_channels = 3
    out_channels = 2
    kernel_size = (2)
    groups = 1
    stride = 1
    padding = [0]
    output_padding = 0
    dilation = 1
    res = np.array(
        [
            [[1.5298, 2.4251, 0.895],[1.5298, 2.4251, 0.895]],
            [[1.6651, 3.4983, 1.833],[1.6651, 3.4984, 1.834]],
        ]
    )

    obj.run(
        res=res,
        data=x,
        in_channels=in_channels,
        out_channels=out_channels,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        kernel_size = kernel_size,
        groups=groups,
        dilation=dilation,
        weight_attr=fluid.initializer.ConstantInitializer(value=1),
        bias_attr=fluid.initializer.ConstantInitializer(value=0),
        data_format="NCL",
    )

@pytest.mark.api_nn_Conv1D_Transpose_parameters
def test_conv1d_transpose3():
    """
    kernel_size = [2],out_channels = 3
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 2])
    in_channels = 3
    out_channels = 3
    kernel_size = [2]
    groups = 1
    stride = 1
    padding = 0
    output_padding = 0
    dilation = 1
    res = np.array(
        [
            [
                 [1.5299, 2.4252, 0.8954],
                 [1.5297, 2.4252, 0.8950],
                 [1.5295, 2.4251, 0.8956]
            ],
            [
                 [1.6651, 3.4983, 1.8332],
                 [1.6652, 3.4983, 1.8332],
                 [1.6651, 3.4983, 1.8332]
            ]
        ]
    )

    obj.run(
        res=res,
        data=x,
        in_channels=in_channels,
        out_channels=out_channels,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        kernel_size = kernel_size,
        groups=groups,
        dilation=dilation,
        weight_attr=fluid.initializer.ConstantInitializer(value=1),
        bias_attr=fluid.initializer.ConstantInitializer(value=0),
        data_format="NCL",
    )

@pytest.mark.api_nn_Conv1D_Transpose_parameters
def test_conv1d_transpose4():
    """
    kernel_size = [3] , out_channels = 3 , stride = 2
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 2])
    in_channels = 3
    out_channels = 3
    kernel_size = [3]
    stride = 2
    padding = 0
    res = np.array(
        [
            [[1.5298465, 1.5298465, 2.4251616, 0.89531505, 0.89531505],
             [1.5298465, 1.5298465, 2.4251616, 0.89531505, 0.89531505],
             [1.5298465, 1.5298465, 2.4251616, 0.89531505, 0.89531505]],
            [[1.665139, 1.665139, 3.4983778, 1.8332388, 1.8332388],
             [1.665139,  1.665139,  3.4983778, 1.8332388, 1.8332388],
             [1.665139, 1.665139, 3.4983778, 1.8332388, 1.8332388]]
        ]
    )

    obj.run(
        res=res,
        data=x,
        in_channels=in_channels,
        out_channels=out_channels,
        stride=stride,
        padding=padding,
        kernel_size=kernel_size,
        weight_attr=fluid.initializer.ConstantInitializer(value=1),
        bias_attr=fluid.initializer.ConstantInitializer(value=0),
        data_format="NCL",
    )


@pytest.mark.api_nn_Conv1D_Transpose_parameters
def test_conv1d_transpose5():
    """
    kernel_size = [3] stride = [2] padding=1
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 2])
    in_channels = 3
    out_channels = 1
    kernel_size = [3]
    stride = [2]
    padding = 1
    res = np.array(
        [
            [[1.5298465, 2.4251616, 0.89531505]],
            [[1.665139, 3.4983778, 1.8332388]]
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
        weight_attr=fluid.initializer.ConstantInitializer(value=1),
        bias_attr=fluid.initializer.ConstantInitializer(value=0),
    )


@pytest.mark.api_nn_Conv1D_Transpose_parameters
def test_conv1d_transpose6():
    """
    kernel_size = [3] stride = (2) padding=(0) groups=3
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 2])
    in_channels = 3
    out_channels = 6
    kernel_size = [3]
    stride = (2)
    padding = (0)
    groups = 3

    res = np.array(
        [
            [
                [0.24851012, 0.24851012, 0.69848555, 0.44997543, 0.44997543],
                [0.24851012, 0.24851012, 0.69848555, 0.44997543, 0.44997543],
                [0.4109408, 0.4109408, 0.67124045, 0.26029968, 0.26029968],
                [0.4109408, 0.4109408, 0.67124045, 0.26029968, 0.26029968],
                [0.87039566, 0.87039566, 1.0554355, 0.18503992, 0.18503992],
                [0.87039566, 0.87039566, 1.0554355, 0.18503992, 0.18503992],
            ],
            [
                [0.01966142, 0.01966142, 0.97291344, 0.953252, 0.953252],
                [0.01966142, 0.01966142, 0.97291344, 0.953252, 0.953252],
                [0.6804508, 0.6804508, 1.1670389, 0.48658812, 0.48658812],
                [0.6804508, 0.6804508, 1.1670389, 0.48658812, 0.48658812],
                [0.9650268, 0.9650268, 1.3584255, 0.39339873, 0.39339873],
                [0.9650268, 0.9650268, 1.3584255, 0.39339873, 0.39339873]
            ]
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
        groups=groups,
        weight_attr=fluid.initializer.ConstantInitializer(value=1),
        bias_attr=fluid.initializer.ConstantInitializer(value=0),
    )


@pytest.mark.api_nn_Conv1D_Transpose_parameters
def test_conv1d_transpose7():
    """
    kernel_size = [3] stride = 1 padding=[0] w=0.7 b=-0.3
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 2])
    in_channels = 3
    out_channels = 2
    kernel_size = [3]
    stride = 1
    padding = 0
    res = np.array(
        [
            [[0.77089256, 1.397613, 1.397613, 0.32672048],
             [0.77089256, 1.397613, 1.397613, 0.32672048]],
             [[0.8655973, 2.1488645, 2.1488645, 0.9832671],
             [0.8655973, 2.1488645, 2.1488645, 0.9832671]]
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
        weight_attr=fluid.initializer.ConstantInitializer(value=0.7),
        bias_attr=fluid.initializer.ConstantInitializer(value=-0.3),
    )


@pytest.mark.api_nn_Conv1D_Transpose_parameters
def test_conv1d_transpose8():
    """
    kernel_size = [3] stride = 1 padding=1 w=0.7 b=-0.3 data_format="NLC"
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 2]).transpose(0, 2, 1)
    in_channels = 3
    out_channels = 2
    kernel_size = [3]
    stride = 1
    padding = 0
    data_format = "NLC"
    res = np.array(
        [
            [[0.77089256, 1.397613, 1.397613, 0.32672048],
             [0.77089256, 1.397613, 1.397613, 0.32672048]],

            [[0.8655973, 2.1488645, 2.1488645, 0.9832671],
             [0.8655973, 2.1488645, 2.1488645, 0.9832671]]
        ]
    ).transpose(0, 2, 1)
    obj.run(
        res=res,
        data=x,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        data_format=data_format,
        weight_attr=fluid.initializer.ConstantInitializer(value=0.7),
        bias_attr=fluid.initializer.ConstantInitializer(value=-0.3),
    )


@pytest.mark.api_nn_Conv1D_Transpose_parameters
def test_conv1d_transpose9():
    """
    padding = "VALID"
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 2])
    in_channels = 3
    out_channels = 1
    kernel_size = [3]
    stride = 1
    padding = "VALID"
    res = np.array(
        [
            [[0.77089256, 1.397613, 1.397613, 0.32672048]],
            [[0.8655973, 2.1488645, 2.1488645, 0.9832671]]
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
        weight_attr=fluid.initializer.ConstantInitializer(value=0.7),
        bias_attr=fluid.initializer.ConstantInitializer(value=-0.3),
    )


@pytest.mark.api_nn_Conv1D_Transpose_parameters
def test_conv1d_transpose10():
    """
    padding = "SAME"
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4])
    in_channels = 3
    out_channels = 1
    kernel_size = [3]
    stride = 1
    padding = "SAME"
    res = np.array(
        [
            [[1.7446721, 2.7216122, 2.5869281, 1.8018057]],
             [[1.6420112, 2.4731734, 2.4002645, 1.6349971]]
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
        weight_attr=fluid.initializer.ConstantInitializer(value=0.7),
        bias_attr=fluid.initializer.ConstantInitializer(value=-0.3),
    )


@pytest.mark.api_nn_Conv1D_Transpose_parameters
def test_conv1d_transpose11():
    """
    padding="SAME"  dilation = (2)
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4])
    in_channels = 3
    out_channels = 1
    kernel_size = [3]
    stride = 1
    padding = "SAME"
    dilation = (2)
    res = np.array(
        [
            [[1.7446721, 2.7216122, 2.5869281, 1.8018057]],
             [[1.6420112, 2.4731734, 2.4002645, 1.6349971]]
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
        dilation = dilation,
        weight_attr=fluid.initializer.ConstantInitializer(value=0.7),
        bias_attr=fluid.initializer.ConstantInitializer(value=-0.3),
    )


@pytest.mark.api_nn_Conv1D_Transpose_parameters
def test_conv1d_transpose12():
    """
    padding = "VALID" dilation = [2]
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4])
    in_channels = 3
    out_channels = 1
    kernel_size = [3]
    stride = 1
    padding = "VALID"
    dilation = [2]

    res = np.array(
        [
            [[0.9595495, 0.48512244, 1.9364898, 1.6099877, 1.9364898, 1.6099877, 0.67694026, 0.8248653]],
            [[0.87674373, 0.46526742, 1.707906, 1.5691025, 1.707906, 1.5691025, 0.5311621, 0.803835]]
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
        weight_attr=fluid.initializer.ConstantInitializer(value=0.7),
        bias_attr=fluid.initializer.ConstantInitializer(value=-0.3),
    )


@pytest.mark.api_nn_Conv1D_Transpose_parameters
def test_conv1d_transpose13():
    """
    dilation = (2,)
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4])
    in_channels = 3
    out_channels = 1
    kernel_size = [3]
    stride = 1
    padding = 1
    dilation = (2,)
    res = np.array([[[0.48512244, 1.9364898, 1.6099877, 1.9364898, 1.6099877, 0.67694026]],

                    [[0.46526742,1.707906,  1.5691025, 1.707906,  1.5691025, 0.5311621]]])
    obj.run(
        res=res,
        data=x,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        weight_attr=fluid.initializer.ConstantInitializer(value=0.7),
        bias_attr=fluid.initializer.ConstantInitializer(value=-0.3),
    )



@pytest.mark.api_nn_Conv1D_Transpose_parameters
def test_conv1d_transpose14():
    """
     dilation = (2, 2)  padding = (1, 2)
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 2])
    in_channels = 3
    out_channels = 1
    kernel_size = [3]
    stride = 1
    padding = (1,)
    dilation = (2,)

    res = np.array([[[0.32672048,0.77089256, 0.32672048, 0.77089256]],
                    [[0.9832671, 0.8655973, 0.9832671, 0.8655973]]])
    obj.run(
        res=res,
        data=x,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        weight_attr=fluid.initializer.ConstantInitializer(value=0.7),
        bias_attr=fluid.initializer.ConstantInitializer(value=-0.3),
    )

