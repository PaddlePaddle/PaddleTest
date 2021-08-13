#!/usr/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python

"""
Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
File: test_prune.py

Authors:
Date: 2020/10/27 10:02
"""

# Copyright (c) 2019  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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
import sys
import unittest
import paddle.fluid as fluid
from paddleslim.prune import Pruner
from static_case import StaticCase
from layers import conv_bn_layer
import paddle

sys.path.append("../")

if paddle.is_compiled_with_cuda() is True:
    # places = [paddle.CPUPlace(), paddle.CUDAPlace(0)]
    places = [paddle.CPUPlace()]
else:
    # default
    places = [paddle.CPUPlace()]


class TestPrune(StaticCase):
    """
    test paddleslim.prune.Pruner.prune(program, scope, params, ratios, place=None, lazy=False, only_graph=False, \
    param_backup=False, param_shape_backup=False)
    """

    def test_prune1(self):
        """
        criterion="l1_norm",lazy=False,only_graph=False
        :return:
        """
        main_program = fluid.Program()
        startup_program = fluid.Program()
        #   X       X              O       X              O
        # conv1-->conv2-->sum1-->conv3-->conv4-->sum2-->conv5-->conv6
        #     |            ^ |                    ^
        #     |____________| |____________________|
        #
        # X: prune output channels
        # O: prune input channels
        with fluid.program_guard(main_program, startup_program):
            input = fluid.data(name="image", shape=[None, 3, 16, 16])
            conv1 = conv_bn_layer(input, 8, 3, "conv1")
            conv2 = conv_bn_layer(conv1, 8, 3, "conv2")
            sum1 = conv1 + conv2
            conv3 = conv_bn_layer(sum1, 8, 3, "conv3")
            conv4 = conv_bn_layer(conv3, 8, 3, "conv4")
            sum2 = conv4 + sum1
            conv5 = conv_bn_layer(sum2, 8, 3, "conv5")
            conv6 = conv_bn_layer(conv5, 8, 3, "conv6")

            fluid.layers.conv2d_transpose(input=conv6, num_filters=16, filter_size=2, stride=2)

        shapes = {}
        for param in main_program.global_block().all_parameters():
            shapes[param.name] = param.shape

        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        scope = fluid.Scope()
        exe.run(startup_program, scope=scope)
        pruner = Pruner(criterion="l1_norm")
        main_program, _, _ = pruner.prune(
            main_program,
            scope,
            params=["conv4_weights", "conv2d_transpose_0.w_0"],
            ratios=[0.5, 0.6],
            place=place,
            lazy=False,
            only_graph=False,
            param_backup=None,
            param_shape_backup=None,
        )

        shapes = {
            "conv1_weights": (4, 3, 3, 3),
            "conv2_weights": (4, 4, 3, 3),
            "conv3_weights": (8, 4, 3, 3),
            "conv4_weights": (4, 8, 3, 3),
            "conv5_weights": (8, 4, 3, 3),
            "conv6_weights": (8, 8, 3, 3),
            "conv2d_transpose_0.w_0": (8, 16, 2, 2),
        }

        for param in main_program.global_block().all_parameters():
            if param.name in shapes:
                print("param: {}; param shape: {}".format(param.name, param.shape))
                self.assertTrue(param.shape == shapes[param.name])

    def test_prune2(self):
        """
        criterion="bn_scale",lazy=True,only_graph=True,param_backup=True,param_shape_backup=True
        :return:
        """

        main_program = fluid.Program()
        startup_program = fluid.Program()
        #   X       X              O       X              O
        # conv1-->conv2-->sum1-->conv3-->conv4-->sum2-->conv5-->conv6
        #     |            ^ |                    ^
        #     |____________| |____________________|
        #
        # X: prune output channels
        # O: prune input channels
        with fluid.program_guard(main_program, startup_program):
            input = fluid.data(name="image", shape=[None, 3, 16, 16])
            conv1 = conv_bn_layer(input, 8, 3, "conv1")
            conv2 = conv_bn_layer(conv1, 8, 3, "conv2")
            sum1 = conv1 + conv2
            conv3 = conv_bn_layer(sum1, 8, 3, "conv3")
            conv4 = conv_bn_layer(conv3, 8, 3, "conv4")
            sum2 = conv4 + sum1
            conv5 = conv_bn_layer(sum2, 8, 3, "conv5")
            conv6 = conv_bn_layer(conv5, 8, 3, "conv6")

            fluid.layers.conv2d_transpose(input=conv6, num_filters=16, filter_size=2, stride=2)

        shapes = {}
        for param in main_program.global_block().all_parameters():
            shapes[param.name] = param.shape

        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        scope = fluid.Scope()
        exe.run(startup_program, scope=scope)
        pruner = Pruner(criterion="bn_scale")
        main_program, _, _ = pruner.prune(
            main_program,
            scope,
            params=["conv4_weights", "conv2d_transpose_0.w_0"],
            ratios=[0.5, 0.6],
            place=place,
            lazy=True,
            only_graph=True,
            param_backup=True,
            param_shape_backup=True,
        )

        shapes = {
            "conv1_weights": (8, 3, 3, 3),
            "conv2_weights": (8, 8, 3, 3),
            "conv3_weights": (8, 8, 3, 3),
            "conv4_weights": (8, 8, 3, 3),
            "conv5_weights": (8, 8, 3, 3),
            "conv6_weights": (8, 8, 3, 3),
            "conv2d_transpose_0.w_0": (8, 16, 2, 2),
        }

        for param in main_program.global_block().all_parameters():
            if param.name in shapes:
                print("param: {}; param shape: {}".format(param.name, param.shape))
                self.assertTrue(param.shape == shapes[param.name])

    def test_prune3(self):
        """
        criterion="geometry_median",lazy=True,only_graph=True
        :return:
        """

        main_program = fluid.Program()
        startup_program = fluid.Program()
        #   X       X              O       X              O
        # conv1-->conv2-->sum1-->conv3-->conv4-->sum2-->conv5-->conv6
        #     |            ^ |                    ^
        #     |____________| |____________________|
        #
        # X: prune output channels
        # O: prune input channels
        with fluid.program_guard(main_program, startup_program):
            input = fluid.data(name="image", shape=[None, 3, 16, 16])
            conv1 = conv_bn_layer(input, 8, 3, "conv1")
            conv2 = conv_bn_layer(conv1, 8, 3, "conv2")
            sum1 = conv1 + conv2
            conv3 = conv_bn_layer(sum1, 8, 3, "conv3")
            conv4 = conv_bn_layer(conv3, 8, 3, "conv4")
            sum2 = conv4 + sum1
            conv5 = conv_bn_layer(sum2, 8, 3, "conv5")
            conv6 = conv_bn_layer(conv5, 8, 3, "conv6")

            fluid.layers.conv2d_transpose(input=conv6, num_filters=16, filter_size=2, stride=2)

        shapes = {}
        for param in main_program.global_block().all_parameters():
            shapes[param.name] = param.shape

        place = fluid.CPUPlace()
        # place = fluid.CUDAPlace(0)
        exe = fluid.Executor(place)
        scope = fluid.Scope()
        exe.run(startup_program, scope=scope)
        pruner = Pruner(criterion="geometry_median")
        main_program, _, _ = pruner.prune(
            main_program,
            scope,
            params=["conv4_weights", "conv2d_transpose_0.w_0"],
            ratios=[0.5, 0.6],
            place=place,
            lazy=True,
            only_graph=True,
            param_backup=False,
            param_shape_backup=False,
        )

        shapes = {
            "conv1_weights": (8, 3, 3, 3),
            "conv2_weights": (8, 8, 3, 3),
            "conv3_weights": (8, 8, 3, 3),
            "conv4_weights": (8, 8, 3, 3),
            "conv5_weights": (8, 8, 3, 3),
            "conv6_weights": (8, 8, 3, 3),
            "conv2d_transpose_0.w_0": (8, 16, 2, 2),
        }

        for param in main_program.global_block().all_parameters():
            if param.name in shapes:
                print("param: {}; param shape: {}".format(param.name, param.shape))
                self.assertTrue(param.shape == shapes[param.name])


if __name__ == "__main__":
    unittest.main()
