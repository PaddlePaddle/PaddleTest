#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python

# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
            [
                [[9.148826, 7.6043386], [9.538667, 8.41972]],
                [[5.923057, 4.366275], [5.420381, 5.990474]],
            ],
            [
                [[9.90173, 8.549506], [11.014879, 9.594643]],
                [[6.388487, 6.04265], [6.0694923, 6.03674]],
            ],
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
        weight_attr=paddle.nn.initializer.Assign(
            randtool("float", 0, 1, [2, 3, 3, 3])
        ),
        bias_attr=paddle.nn.initializer.Assign(randtool("float", 0, 1, [2])),
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
                [
                    [
                        3.231795,
                        2.486073,
                        2.282403,
                    ],
                    [2.7349014, 2.943695, 3.2125316],
                    [
                        2.891021,
                        3.84766,
                        4.053831,
                    ],
                ],
                [
                    [3.1089513, 2.0693536, 2.8869987],
                    [2.6687198, 2.8988774, 3.1295364],
                    [2.4657903, 3.2080243, 4.2252774],
                ],
            ],
            [
                [
                    [2.6912615, 2.9456987, 3.6711123],
                    [4.552159, 2.8042006, 3.7762904],
                    [4.0862513, 3.8355634, 3.4726915],
                ],
                [
                    [2.8221548, 2.5010462, 3.2962692],
                    [3.8097546, 2.869685, 3.2989302],
                    [3.8046136, 3.5966544, 3.27493],
                ],
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

    b_init = [0.33147673, 0.3539822]
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
            [
                [[9.148826, 7.6043386], [9.538667, 8.41972]],
                [[5.923057, 4.366275], [5.420381, 5.990474]],
            ],
            [
                [[9.90173, 8.549506], [11.014879, 9.594643]],
                [[6.388487, 6.04265], [6.0694923, 6.03674]],
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
        weight_attr=paddle.nn.initializer.Assign(
            randtool("float", 0, 1, [2, 3, 3, 3])
        ),
        bias_attr=paddle.nn.initializer.Assign(randtool("float", 0, 1, [2])),
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
    res = np.array(
        [[[[8.935533]], [[9.181023]]], [[[9.6884365]], [[9.933927]]]]
    )
    obj.run(
        res=res,
        data=x,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        weight_attr=paddle.nn.initializer.Assign(
            randtool("float", 0, 1, [1, 3, 3, 3])
        ),
        bias_attr=paddle.nn.initializer.Assign(randtool("float", 0, 1, [2])),
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
    res = np.array([[[[9.148826]], [[5.923057]]], [[[9.90173]], [[6.3884873]]]])
    obj.run(
        res=res,
        data=x,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        weight_attr=paddle.nn.initializer.Assign(
            randtool("float", 0, 1, [2, 3, 3, 3])
        ),
        bias_attr=paddle.nn.initializer.Assign(randtool("float", 0, 1, [2])),
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
    res = np.array(
        [
            [
                [
                    [4.0355253, 5.0900354],
                    [
                        4.9440603,
                        8.206427,
                    ],
                ],
                [
                    [4.2810154, 5.3355255],
                    [
                        5.1895504,
                        8.451917,
                    ],
                ],
            ],
            [
                [
                    [
                        3.761713,
                        5.114859,
                    ],
                    [
                        6.6824675,
                        9.38135,
                    ],
                ],
                [
                    [
                        4.007203,
                        5.360349,
                    ],
                    [6.9279575, 9.62684],
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
        weight_attr=paddle.nn.initializer.Assign(
            randtool("float", 0, 1, [1, 3, 3, 3])
        ),
        bias_attr=paddle.nn.initializer.Assign(randtool("float", 0, 1, [2])),
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
    res = np.array(
        [
            [
                [[4.0355253, 5.0900354], [4.9440603, 8.206427]],
                [[4.2810154, 5.3355255], [5.1895504, 8.451917]],
            ],
            [
                [[3.761713, 5.114859], [6.6824675, 9.38135]],
                [[4.007203, 5.360349], [6.9279575, 9.62684]],
            ],
        ]
    )
    obj.static = False
    obj.run(
        res=res,
        data=x,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        weight_attr=paddle.nn.initializer.Assign(
            paddle.to_tensor(randtool("float", 0, 1, [1, 3, 3, 3]))
        ),
        bias_attr=paddle.nn.initializer.Assign(
            paddle.to_tensor(randtool("float", 0, 1, [2]))
        ),
    )
    obj.static = True
