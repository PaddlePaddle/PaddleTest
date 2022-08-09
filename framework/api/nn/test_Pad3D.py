#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_Pad3D
"""
from apibase import APIBase
from apibase import randtool
import paddle
import pytest
import numpy as np


class TestPad3D(APIBase):
    """
    test
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32, np.float64, np.int32, np.int64]
        # self.debug = True
        # self.static = True
        # enable check grad
        self.no_grad_var = ["padding"]
        self.enable_backward = True


obj = TestPad3D(paddle.nn.Pad3D)


@pytest.mark.api_nn_Pad3D_vartype
def test_Pad3D_base():
    """
    shape_dim=3, pad=list[1, 2], mode='constant', len(pad)=2, data_format=NCL
    """
    input_shape = (1, 2, 3)
    pad = [1, 2]
    mode = "constant"
    res = [[[0, 1, 2, 3, 0, 0], [0, 4, 5, 6, 0, 0]]]
    data = np.arange(np.prod(input_shape)).reshape(input_shape) + 1
    obj.base(res=res, padding=pad, mode=mode, data_format="NCL", data=data)


@pytest.mark.api_nn_Pad3D_parameters
def test_Pad3D1():
    """
    shape_dim=3, pad=list[1, 2], mode='constant', len(pad)=2, data_format=NCL
    """
    input_shape = (1, 2, 3)
    pad = [1, 2]
    mode = "constant"
    res = [[[0, 1, 2, 3, 0, 0], [0, 4, 5, 6, 0, 0]]]
    data = np.arange(np.prod(input_shape)).reshape(input_shape) + 1
    obj.base(res=res, padding=pad, mode=mode, data_format="NCL", data=data)


@pytest.mark.api_nn_Pad3D_parameters
def test_Pad3D2():
    """
    shape_dim=3, pad=list[1, 0, 1, 2], mode='constant', len(pad)=4, data_format=NCHW
    """
    input_shape = (1, 1, 2, 3)
    pad = [1, 0, 1, 2]
    mode = "constant"
    res = [[[[0, 0, 0, 0], [0, 1, 2, 3], [0, 4, 5, 6], [0, 0, 0, 0], [0, 0, 0, 0]]]]
    data = np.arange(np.prod(input_shape)).reshape(input_shape) + 1
    obj.run(res=res, padding=pad, mode=mode, data_format="NCHW", data=data)


@pytest.mark.api_nn_Pad3D_parameters
def test_Pad3D3():
    """
    shape_dim=3, pad=list[1, 0, 1, 2, 1, 0], mode='constant', len(pad)=5, data_format=NCDHW
    """
    input_shape = (1, 1, 2, 3, 2)
    pad = [1, 0, 1, 2, 1, 0]
    mode = "constant"
    res = [
        [
            [
                [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 1, 2], [0, 3, 4], [0, 5, 6], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 7, 8], [0, 9, 10], [0, 11, 12], [0, 0, 0], [0, 0, 0]],
            ]
        ]
    ]
    data = np.arange(np.prod(input_shape)).reshape(input_shape) + 1
    obj.run(res=res, padding=pad, mode=mode, data_format="NCDHW", data=data)


@pytest.mark.api_nn_Pad3D_parameters
def test_Pad3D4():
    """
    shape_dim=3, pad=tensor[1, 2], mode='constant', len(pad)=2, data_format=NCL
    """
    input_shape = (1, 2, 3)
    # pad = np.array([1, 2]).astype('int32')
    pad = [1, 2]
    mode = "constant"
    res = [[[0, 1, 2, 3, 0, 0], [0, 4, 5, 6, 0, 0]]]
    data = np.arange(np.prod(input_shape)).reshape(input_shape) + 1
    obj.base(res=res, padding=pad, mode=mode, data_format="NCL", data=data)


@pytest.mark.api_nn_Pad3D_parameters
def test_Pad3D5():
    """
    shape_dim=3, pad=tensor[1, 0, 1, 2], mode='constant', len(pad)=4, data_format=NCHW
    """
    input_shape = (1, 1, 2, 3)
    # pad = np.array([1, 0, 1, 2]).astype('int32')
    pad = [1, 0, 1, 2]
    mode = "constant"
    res = [[[[0, 0, 0, 0], [0, 1, 2, 3], [0, 4, 5, 6], [0, 0, 0, 0], [0, 0, 0, 0]]]]
    data = np.arange(np.prod(input_shape)).reshape(input_shape) + 1
    obj.run(res=res, padding=pad, mode=mode, data_format="NCHW", data=data)


@pytest.mark.api_nn_Pad3D_parameters
def test_Pad3D6():
    """
    shape_dim=3, pad=tensor[1, 0, 1, 2, 1, 0], mode='constant', len(pad)=5, data_format=NCDHW
    """
    input_shape = (1, 1, 2, 3, 2)
    # pad = np.array([1, 0, 1, 2, 1, 0]).astype('int32')
    pad = [1, 0, 1, 2, 1, 0]
    mode = "constant"
    res = [
        [
            [
                [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 1, 2], [0, 3, 4], [0, 5, 6], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 7, 8], [0, 9, 10], [0, 11, 12], [0, 0, 0], [0, 0, 0]],
            ]
        ]
    ]
    data = np.arange(np.prod(input_shape)).reshape(input_shape) + 1
    obj.run(res=res, padding=pad, mode=mode, data_format="NCDHW", data=data)


@pytest.mark.api_nn_Pad3D_parameters
def test_Pad3D7():
    """
    shape_dim=3, pad=list[1, 2], mode='reflect', len(pad)=2, data_format=NCL
    must set left and right value < W,  top and bottom < H.
    """
    input_shape = (1, 2, 3)
    pad = [1, 2]
    mode = "reflect"
    res = [[[2, 1, 2, 3, 2, 1], [5, 4, 5, 6, 5, 4]]]
    data = np.arange(np.prod(input_shape)).reshape(input_shape) + 1
    obj.base(res=res, padding=pad, mode=mode, data_format="NCL", data=data)


@pytest.mark.api_nn_Pad3D_parameters
def test_Pad3D8():
    """
    shape_dim=3, pad=list[2, 1], mode='reflect', len(pad)=2, data_format=NLC
    """
    input_shape = (1, 2, 3)
    pad = [1, 1]
    mode = "reflect"
    res = [[[4, 5, 6], [1, 2, 3], [4, 5, 6], [1, 2, 3]]]
    data = np.arange(np.prod(input_shape)).reshape(input_shape) + 1
    obj.run(res=res, padding=pad, mode=mode, data_format="NLC", data=data)


@pytest.mark.api_nn_Pad3D_parameters
def test_Pad3D9():
    """
    shape_dim=3, pad=list[1, 2], mode='reflect', len(pad)=2, data_format=NCL
    """
    input_shape = (1, 2, 3)
    pad = [1, 2]
    mode = "reflect"
    res = [[[2, 1, 2, 3, 2, 1], [5, 4, 5, 6, 5, 4]]]
    data = np.arange(np.prod(input_shape)).reshape(input_shape) + 1
    obj.run(res=res, padding=pad, mode=mode, data=data, data_format="NCL")


@pytest.mark.api_nn_Pad3D_parameters
def test_Pad3D10():
    """
    shape_dim=3, pad=list[1, 1, 1, 0], mode='reflect', len(pad)=2, data_format='NHWC'
    """
    input_shape = (1, 2, 3, 1)
    pad = [1, 1, 1, 0]
    mode = "reflect"
    res = [[[[5], [4], [5], [6], [5]], [[2], [1], [2], [3], [2]], [[5], [4], [5], [6], [5]]]]
    data = np.arange(np.prod(input_shape)).reshape(input_shape) + 1
    obj.run(res=res, padding=pad, mode=mode, data_format="NHWC", data=data)


@pytest.mark.api_nn_Pad3D_parameters
def test_Pad3D11():
    """
    shape_dim=3, pad=list[1, 1], mode='reflect', len(pad)=2, data_format=NLC
    """
    input_shape = (1, 2, 3)
    # pad = np.array([1, 1]).astype('int32')
    pad = [1, 1]
    mode = "reflect"
    res = [[[4, 5, 6], [1, 2, 3], [4, 5, 6], [1, 2, 3]]]
    data = np.arange(np.prod(input_shape)).reshape(input_shape) + 1
    obj.run(res=res, padding=pad, mode=mode, data_format="NLC", data=data)


@pytest.mark.api_nn_Pad3D_parameters
def test_Pad3D12():
    """
    shape_dim=3, pad=list[1, 2], mode='reflect', len(pad)=2, data_format=NCL
    """
    input_shape = (1, 2, 3)
    # pad = np.array([1, 2]).astype('int32')
    pad = [1, 2]
    mode = "reflect"
    res = [[[2, 1, 2, 3, 2, 1], [5, 4, 5, 6, 5, 4]]]
    data = np.arange(np.prod(input_shape)).reshape(input_shape) + 1
    obj.run(res=res, padding=pad, mode=mode, data=data, data_format="NCL")


@pytest.mark.api_nn_Pad3D_parameters
def test_Pad3D13():
    """
    shape_dim=3, pad=tensor[1, 1, 1, 0], mode='reflect', len(pad)=2, data_format='NHWC'
    """
    input_shape = (1, 2, 3, 1)
    pad = [1, 1, 1, 0]
    mode = "reflect"
    res = [[[[5], [4], [5], [6], [5]], [[2], [1], [2], [3], [2]], [[5], [4], [5], [6], [5]]]]
    data = np.arange(np.prod(input_shape)).reshape(input_shape) + 1
    obj.run(res=res, padding=pad, mode=mode, data_format="NHWC", data=data)


@pytest.mark.api_nn_Pad3D_parameters
def test_Pad3D14():
    """
    shape_dim=5, pad=tensor[1, 1, 1, 0, 1, 0], mode='reflect', len(pad)=6, data_format='NCDHW'
    """
    input_shape = (1, 1, 2, 2, 3)
    pad = [1, 1, 1, 0, 1, 0]
    mode = "reflect"
    res = [
        [
            [
                [[11, 10, 11, 12, 11], [8, 7, 8, 9, 8], [11, 10, 11, 12, 11]],
                [[5, 4, 5, 6, 5], [2, 1, 2, 3, 2], [5, 4, 5, 6, 5]],
                [[11, 10, 11, 12, 11], [8, 7, 8, 9, 8], [11, 10, 11, 12, 11]],
            ]
        ]
    ]
    data = np.arange(np.prod(input_shape)).reshape(input_shape) + 1
    obj.run(res=res, padding=pad, mode=mode, data_format="NCDHW", data=data)


@pytest.mark.api_nn_Pad3D_parameters
def test_Pad3D15():
    """
    shape_dim=5, pad=tensor[1, 1, 1, 0, 1, 0], mode='reflect', len(pad)=6, data_format='NDHWC'
    """
    input_shape = (1, 2, 2, 2, 2)
    pad = [1, 1, 1, 0, 1, 0]
    mode = "reflect"
    res = [
        [
            [
                [[15, 16], [13, 14], [15, 16], [13, 14]],
                [[11, 12], [9, 10], [11, 12], [9, 10]],
                [[15, 16], [13, 14], [15, 16], [13, 14]],
            ],
            [[[7, 8], [5, 6], [7, 8], [5, 6]], [[3, 4], [1, 2], [3, 4], [1, 2]], [[7, 8], [5, 6], [7, 8], [5, 6]]],
            [
                [[15, 16], [13, 14], [15, 16], [13, 14]],
                [[11, 12], [9, 10], [11, 12], [9, 10]],
                [[15, 16], [13, 14], [15, 16], [13, 14]],
            ],
        ]
    ]
    data = np.arange(np.prod(input_shape)).reshape(input_shape) + 1
    obj.run(res=res, padding=pad, mode=mode, data_format="NDHWC", data=data)


@pytest.mark.api_nn_Pad3D_parameters
def test_Pad3D16():
    """
    shape_dim=3, pad=list[1, 2], mode='replicate', len(pad)=2, data_format=NCL
    """
    input_shape = (1, 2, 3)
    pad = [1, 2]
    mode = "replicate"
    res = [[[1, 1, 2, 3, 3, 3], [4, 4, 5, 6, 6, 6]]]
    data = np.arange(np.prod(input_shape)).reshape(input_shape) + 1
    obj.base(res=res, padding=pad, mode=mode, data=data, data_format="NCL")


@pytest.mark.api_nn_Pad3D_parameters
def test_Pad3D17():
    """
    shape_dim=3, pad=list[2, 1], mode='replicate', len(pad)=2, data_format=NCL
    """
    input_shape = (1, 2, 3)
    pad = [2, 1]
    mode = "replicate"
    res = [[[1, 1, 1, 2, 3, 3], [4, 4, 4, 5, 6, 6]]]
    data = np.arange(np.prod(input_shape)).reshape(input_shape) + 1
    obj.run(res=res, padding=pad, mode=mode, data=data, data_format="NCL")


@pytest.mark.api_nn_Pad3D_parameters
def test_Pad3D18():
    """
    shape_dim=4, pad=list[2, 1, 2, 1], mode='replicate', len(pad)=4, data_format=NCHW
    """
    input_shape = (1, 2, 3, 4)
    pad = [2, 1, 2, 1]
    mode = "replicate"
    res = [
        [
            [
                [1, 1, 1, 2, 3, 4, 4],
                [1, 1, 1, 2, 3, 4, 4],
                [1, 1, 1, 2, 3, 4, 4],
                [5, 5, 5, 6, 7, 8, 8],
                [9, 9, 9, 10, 11, 12, 12],
                [9, 9, 9, 10, 11, 12, 12],
            ],
            [
                [13, 13, 13, 14, 15, 16, 16],
                [13, 13, 13, 14, 15, 16, 16],
                [13, 13, 13, 14, 15, 16, 16],
                [17, 17, 17, 18, 19, 20, 20],
                [21, 21, 21, 22, 23, 24, 24],
                [21, 21, 21, 22, 23, 24, 24],
            ],
        ]
    ]
    data = np.arange(np.prod(input_shape)).reshape(input_shape) + 1
    obj.run(res=res, padding=pad, mode=mode, data=data, data_format="NCHW")


@pytest.mark.api_nn_Pad3D_parameters
def test_Pad3D19():
    """
    shape_dim=2, pad=list[1, 2], mode='replicate', len(pad)=2, data_format=NLC
    """
    input_shape = (1, 2, 3)
    pad = [1, 2]
    mode = "replicate"
    res = [[[1, 2, 3], [1, 2, 3], [4, 5, 6], [4, 5, 6], [4, 5, 6]]]
    data = np.arange(np.prod(input_shape)).reshape(input_shape) + 1
    obj.run(res=res, padding=pad, mode=mode, data_format="NLC", data=data)


@pytest.mark.api_nn_Pad3D_parameters
def test_Pad3D20():
    """
    shape_dim=4, pad=list[2, 1, 2, 1], mode='replicate', len(pad)=4, data_format=NCHW
    """
    input_shape = (1, 2, 3, 4)
    pad = [2, 1, 2, 1]
    mode = "replicate"
    res = [
        [
            [
                [1, 1, 1, 2, 3, 4, 4],
                [1, 1, 1, 2, 3, 4, 4],
                [1, 1, 1, 2, 3, 4, 4],
                [5, 5, 5, 6, 7, 8, 8],
                [9, 9, 9, 10, 11, 12, 12],
                [9, 9, 9, 10, 11, 12, 12],
            ],
            [
                [13, 13, 13, 14, 15, 16, 16],
                [13, 13, 13, 14, 15, 16, 16],
                [13, 13, 13, 14, 15, 16, 16],
                [17, 17, 17, 18, 19, 20, 20],
                [21, 21, 21, 22, 23, 24, 24],
                [21, 21, 21, 22, 23, 24, 24],
            ],
        ]
    ]
    data = np.arange(np.prod(input_shape)).reshape(input_shape) + 1
    obj.run(res=res, padding=pad, mode=mode, data_format="NCHW", data=data)


@pytest.mark.api_nn_Pad3D_parameters
def test_Pad3D21():
    """
    shape_dim=3, pad=tensor[2, 1], mode='replicate', len(pad)=2, data_format=NCL
    """
    input_shape = (1, 2, 3)
    # pad = np.array([2, 1]).astype('int32')
    pad = [2, 1]
    mode = "replicate"
    res = [[[1, 1, 1, 2, 3, 3], [4, 4, 4, 5, 6, 6]]]
    data = np.arange(np.prod(input_shape)).reshape(input_shape) + 1
    obj.run(res=res, padding=pad, mode=mode, data=data, data_format="NCL")


@pytest.mark.api_nn_Pad3D_parameters
def test_Pad3D22():
    """
    shape_dim=3, pad=tensor[2, 1, 2, 1], mode='replicate', len(pad)=4, data_format=NCHW
    """
    input_shape = (1, 2, 3, 4)
    # pad = np.array([2, 1, 2, 1]).astype('int32')
    pad = [2, 1, 2, 1]
    mode = "replicate"
    res = [
        [
            [
                [1, 1, 1, 2, 3, 4, 4],
                [1, 1, 1, 2, 3, 4, 4],
                [1, 1, 1, 2, 3, 4, 4],
                [5, 5, 5, 6, 7, 8, 8],
                [9, 9, 9, 10, 11, 12, 12],
                [9, 9, 9, 10, 11, 12, 12],
            ],
            [
                [13, 13, 13, 14, 15, 16, 16],
                [13, 13, 13, 14, 15, 16, 16],
                [13, 13, 13, 14, 15, 16, 16],
                [17, 17, 17, 18, 19, 20, 20],
                [21, 21, 21, 22, 23, 24, 24],
                [21, 21, 21, 22, 23, 24, 24],
            ],
        ]
    ]
    data = np.arange(np.prod(input_shape)).reshape(input_shape) + 1
    obj.run(res=res, padding=pad, mode=mode, data=data, data_format="NCHW")


@pytest.mark.api_nn_Pad3D_parameters
def test_Pad3D23():
    """
    shape_dim=3, pad=tensor[1, 2], mode='replicate', len(pad)=2, data_format=NLC
    """
    input_shape = (1, 2, 3)
    # pad = np.array([1, 2]).astype('int32')
    pad = [1, 2]
    mode = "replicate"
    res = [[[1, 2, 3], [1, 2, 3], [4, 5, 6], [4, 5, 6], [4, 5, 6]]]
    data = np.arange(np.prod(input_shape)).reshape(input_shape) + 1
    obj.run(res=res, padding=pad, mode=mode, data_format="NLC", data=data)


@pytest.mark.api_nn_Pad3D_parameters
def test_Pad3D24():
    """
    shape_dim=4, pad=tensor[2, 1, 2, 1], mode='replicate', len(pad)=4, data_format=NHWC
    """
    input_shape = (1, 2, 3, 4)
    # pad = np.array([2, 1, 2, 1]).astype('int32')
    pad = [2, 1, 2, 1]
    mode = "replicate"
    res = [
        [
            [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [9, 10, 11, 12]],
            [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [9, 10, 11, 12]],
            [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [9, 10, 11, 12]],
            [
                [13, 14, 15, 16],
                [13, 14, 15, 16],
                [13, 14, 15, 16],
                [17, 18, 19, 20],
                [21, 22, 23, 24],
                [21, 22, 23, 24],
            ],
            [
                [13, 14, 15, 16],
                [13, 14, 15, 16],
                [13, 14, 15, 16],
                [17, 18, 19, 20],
                [21, 22, 23, 24],
                [21, 22, 23, 24],
            ],
        ]
    ]
    data = np.arange(np.prod(input_shape)).reshape(input_shape) + 1
    obj.run(res=res, padding=pad, mode=mode, data_format="NHWC", data=data)


@pytest.mark.api_nn_Pad3D_parameters
def test_Pad3D25():
    """
    shape_dim=5, pad=tensor[1, 1, 2, 3, 2], mode='replicate', len(pad)=6, data_format=NCDHW
    """
    input_shape = (1, 1, 2, 3, 2)
    # pad = np.array([1, 0, 1, 0, 0, 1]).astype('int32')
    pad = [1, 0, 1, 0, 0, 1]
    mode = "replicate"
    res = [
        [
            [
                [[1, 1, 2], [1, 1, 2], [3, 3, 4], [5, 5, 6]],
                [[7, 7, 8], [7, 7, 8], [9, 9, 10], [11, 11, 12]],
                [[7, 7, 8], [7, 7, 8], [9, 9, 10], [11, 11, 12]],
            ]
        ]
    ]
    data = np.arange(np.prod(input_shape)).reshape(input_shape) + 1
    obj.run(res=res, padding=pad, mode=mode, data_format="NCDHW", data=data)


@pytest.mark.api_nn_Pad3D_parameters
def test_Pad3D26():
    """
    shape_dim=5, pad=tensor[1, 0, 1, 0, 0, 1], mode='replicate', len(pad)=4, data_format=NDHWC
    """
    input_shape = (1, 1, 2, 3, 2)
    # pad = np.array([1, 0, 1, 0, 0, 1]).astype('int32')
    pad = [1, 0, 1, 0, 0, 1]
    mode = "replicate"
    res = [
        [
            [[[1, 2], [1, 2], [3, 4], [5, 6]], [[1, 2], [1, 2], [3, 4], [5, 6]], [[7, 8], [7, 8], [9, 10], [11, 12]]],
            [[[1, 2], [1, 2], [3, 4], [5, 6]], [[1, 2], [1, 2], [3, 4], [5, 6]], [[7, 8], [7, 8], [9, 10], [11, 12]]],
        ]
    ]
    data = np.arange(np.prod(input_shape)).reshape(input_shape) + 1
    obj.run(res=res, padding=pad, mode=mode, data_format="NDHWC", data=data)


@pytest.mark.api_nn_Pad3D_parameters
def test_Pad3D27():
    """
    case #7
    shape_dim=5, pad=int(2), mode = 'replicate', data_format=NCDHW
    must set left and right value < W,  top and bottom < H.
    """
    input_shape = (1, 1, 2, 2, 2)
    pad = 2
    mode = "replicate"
    res = [
        [
            [
                [
                    [1, 1, 1, 2, 2, 2],
                    [1, 1, 1, 2, 2, 2],
                    [1, 1, 1, 2, 2, 2],
                    [3, 3, 3, 4, 4, 4],
                    [3, 3, 3, 4, 4, 4],
                    [3, 3, 3, 4, 4, 4],
                ],
                [
                    [1, 1, 1, 2, 2, 2],
                    [1, 1, 1, 2, 2, 2],
                    [1, 1, 1, 2, 2, 2],
                    [3, 3, 3, 4, 4, 4],
                    [3, 3, 3, 4, 4, 4],
                    [3, 3, 3, 4, 4, 4],
                ],
                [
                    [1, 1, 1, 2, 2, 2],
                    [1, 1, 1, 2, 2, 2],
                    [1, 1, 1, 2, 2, 2],
                    [3, 3, 3, 4, 4, 4],
                    [3, 3, 3, 4, 4, 4],
                    [3, 3, 3, 4, 4, 4],
                ],
                [
                    [5, 5, 5, 6, 6, 6],
                    [5, 5, 5, 6, 6, 6],
                    [5, 5, 5, 6, 6, 6],
                    [7, 7, 7, 8, 8, 8],
                    [7, 7, 7, 8, 8, 8],
                    [7, 7, 7, 8, 8, 8],
                ],
                [
                    [5, 5, 5, 6, 6, 6],
                    [5, 5, 5, 6, 6, 6],
                    [5, 5, 5, 6, 6, 6],
                    [7, 7, 7, 8, 8, 8],
                    [7, 7, 7, 8, 8, 8],
                    [7, 7, 7, 8, 8, 8],
                ],
                [
                    [5, 5, 5, 6, 6, 6],
                    [5, 5, 5, 6, 6, 6],
                    [5, 5, 5, 6, 6, 6],
                    [7, 7, 7, 8, 8, 8],
                    [7, 7, 7, 8, 8, 8],
                    [7, 7, 7, 8, 8, 8],
                ],
            ]
        ]
    ]

    data = np.arange(np.prod(input_shape)).reshape(input_shape) + 1
    obj.base(res=res, padding=pad, mode=mode, data_format="NCDHW", data=data)
