#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_initializer_assign
"""
from apibase import APIBase
from apibase import randtool
import pytest
import paddle
import numpy as np


class TestinitializerAssign(APIBase):
    """
    test
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32]
        self.delta = 1e-3 * 5


obj = TestinitializerAssign(paddle.nn.Conv2D)


@pytest.mark.api_initializer_assign_vartype
def test_initializer_assign_base():
    """
    base
    """
    x = randtool("float", 0, 1, [2, 3, 4, 4])
    in_channels = 3
    out_channels = 1
    kernel_size = [3, 3]
    stride = 1
    padding = 0
    res = np.array(
        [
            [[[9.148826, 7.149759], [9.689443, 7.829164]], [[6.4570565, 4.66857], [5.920138, 6.5285373]]],
            [[[9.529301, 8.021168], [10.5367985, 9.277258]], [[6.9496613, 6.710565], [6.379459, 6.5709414]]],
        ]
    )
    obj.base(
        res=res,
        data=x,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        weight_attr=paddle.nn.initializer.Assign(randtool("float", 0, 1, [2, 3, 3, 3])),
        bias_attr=paddle.nn.initializer.Assign(randtool("float", 0, 1, [2, 2, 2, 2])),
    )


@pytest.mark.api_initializer_assign_parameters
def test_initializer_assign1():
    """
    paddle.nn.initializer.Assign(list)
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4, 4])
    in_channels = 3
    out_channels = 1
    kernel_size = [2, 2]
    stride = 1
    padding = 0
    res = np.array(
        [
            [
                [[3.1488285, 2.6045718, 2.3618672], [2.6637244, 3.482614, 3.0660949], [2.5792058, 4.469435, 4.402805]],
                [[3.2415571, 2.680398, 2.9264152], [2.3942952, 2.8963027, 2.9391894], [3.094975, 3.7346702, 4.365359]],
            ],
            [
                [[2.760744, 3.0655136, 4.0605125], [4.4684505, 3.0955038, 3.5872626], [3.9559507, 3.5853045, 4.094687]],
                [[2.5239108, 2.7466004, 3.6652846], [4.426062, 3.3372722, 3.472499], [3.782108, 3.5966544, 2.999978]],
            ],
        ]
    )
    w_init = [
        [
            [[0.24851013, 0.44997542], [0.4109408, 0.26029969]],
            [[0.87039569, 0.18503993], [0.01966143, 0.95325203]],
            [[0.6804508, 0.48658813], [0.96502682, 0.39339874]],
        ],
        [
            [[0.07955757, 0.35140742], [0.16363516, 0.98316682]],
            [[0.88062818, 0.49406347], [0.40095924, 0.45129146]],
            [[0.72087685, 0.24776828], [0.62277995, 0.14244882]],
        ],
    ]

    b_init = [
        [
            [
                [0.24851013, 0.44997542, 0.4109408],
                [0.26029969, 0.87039569, 0.18503993],
                [0.01966143, 0.95325203, 0.6804508],
            ],
            [
                [0.48658813, 0.96502682, 0.39339874],
                [0.07955757, 0.35140742, 0.16363516],
                [0.98316682, 0.88062818, 0.49406347],
            ],
        ],
        [
            [
                [0.40095924, 0.45129146, 0.72087685],
                [0.24776828, 0.62277995, 0.14244882],
                [0.20117628, 0.08121773, 0.95347229],
            ],
            [
                [0.05573827, 0.59953648, 0.72299763],
                [0.97028972, 0.82156946, 0.52755107],
                [0.33147673, 0.3539822, 0.0790303],
            ],
        ],
    ]
    obj.run(
        res=res,
        data=x,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        weight_attr=paddle.nn.initializer.Assign(w_init),
        bias_attr=paddle.nn.initializer.Assign(b_init),
    )


@pytest.mark.api_initializer_assign_parameters
def test_initializer_assign2():
    """
    kernel_size = [2, 2], out_channels = 3
    paddle.nn.initializer.Assign(np.ndarray)
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4, 4])
    in_channels = 3
    out_channels = 3
    kernel_size = [2, 2]
    stride = 1
    padding = 0
    res = np.array(
        [
            [[[9.148826, 7.149759], [9.689443, 7.829164]], [[6.4570565, 4.66857], [5.920138, 6.5285373]]],
            [[[9.529301, 8.021168], [10.5367985, 9.277258]], [[6.9496613, 6.710565], [6.379459, 6.5709414]]],
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
        weight_attr=paddle.nn.initializer.Assign(randtool("float", 0, 1, [2, 3, 3, 3])),
        bias_attr=paddle.nn.initializer.Assign(randtool("float", 0, 1, [2, 2, 2, 2])),
    )


@pytest.mark.api_initializer_assign_parameters
def test_initializer_assign3():
    """
    kernel_size = [2, 2] stride = 2
    paddle.nn.initializer.Assign(no.ndarry)
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4, 4])
    in_channels = 3
    out_channels = 1
    kernel_size = [3, 3]
    stride = 2
    padding = 0
    res = np.array([[[[8.935533]]], [[[9.933927]]]])
    obj.run(
        res=res,
        data=x,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        weight_attr=paddle.nn.initializer.Assign(randtool("float", 0, 1, [1, 3, 3, 3])),
        bias_attr=paddle.nn.initializer.Assign(randtool("float", 0, 1, [2, 1, 1, 1])),
    )


@pytest.mark.api_initializer_assign_parameters
def test_initializer_assign3_1():
    """
    kernel_size = [2, 2] stride = 2
    paddle.nn.initializer.Assign(np.ndarry)
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4, 4])
    in_channels = 3
    out_channels = 1
    kernel_size = [3, 3]
    stride = 2
    padding = 0
    res = np.array([[[[9.148826]], [[5.923057]]], [[[10.0525055]], [[6.252511]]]])
    obj.run(
        res=res,
        data=x,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        weight_attr=paddle.nn.initializer.Assign(randtool("float", 0, 1, [2, 3, 3, 3])),
        bias_attr=paddle.nn.initializer.Assign(randtool("float", 0, 1, [2, 2, 1, 1])),
    )


@pytest.mark.api_initializer_assign_parameters
def test_initializer_assign4():
    """
    kernel_size = [2, 2] stride = 2 padding=1
    paddle.nn.initializer.Assign(np.ndarray)
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4, 4])
    in_channels = 3
    out_channels = 1
    kernel_size = [3, 3]
    stride = 2
    padding = 1
    res = np.array([[[[4.0355253, 5.3355255], [5.370474, 8.398205]]], [[[3.683806, 4.7267237], [6.8025727, 9.638547]]]])
    obj.run(
        res=res,
        data=x,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        weight_attr=paddle.nn.initializer.Assign(randtool("float", 0, 1, [1, 3, 3, 3])),
        bias_attr=paddle.nn.initializer.Assign(randtool("float", 0, 1, [2, 1, 2, 2])),
    )


@pytest.mark.api_initializer_assign_parameters
def test_initializer_assign5():
    """
    kernel_size = [2, 2] stride = 2 padding=1
    paddle.nn.initializer.Assign(paddle.Tensor)
    """
    np.random.seed(obj.seed)
    paddle.disable_static()
    x = randtool("float", 0, 1, [2, 3, 4, 4])
    in_channels = 3
    out_channels = 1
    kernel_size = [3, 3]
    stride = 2
    padding = 1
    res = np.array([[[[4.0355253, 5.3355255], [5.370474, 8.398205]]], [[[3.683806, 4.7267237], [6.8025727, 9.638547]]]])
    obj.static = False
    obj.run(
        res=res,
        data=x,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        weight_attr=paddle.nn.initializer.Assign(paddle.to_tensor(randtool("float", 0, 1, [1, 3, 3, 3]))),
        bias_attr=paddle.nn.initializer.Assign(paddle.to_tensor(randtool("float", 0, 1, [2, 1, 2, 2]))),
    )
    obj.static = True
