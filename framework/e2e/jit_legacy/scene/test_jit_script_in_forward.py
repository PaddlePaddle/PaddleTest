#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
a_prac
"""
import os
import shutil
import paddle
import paddle.nn.functional as F
import numpy as np


pwd = os.getcwd()
save_path = os.path.join(pwd, "save_path")
if os.path.exists(save_path):
    shutil.rmtree(save_path)
os.mkdir(os.path.join(pwd, "save_path"))
paddle.seed(103)


class LeNet1(paddle.nn.Layer):
    """
    simple nn layers
    """

    def __init__(self):
        """
        init
        """
        super(LeNet1, self).__init__()
        self.conv1 = paddle.nn.Conv2D(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.max_pool1 = paddle.nn.MaxPool2D(kernel_size=2, stride=2)
        self.conv2 = paddle.nn.Conv2D(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.max_pool2 = paddle.nn.MaxPool2D(kernel_size=2, stride=2)
        self.linear1 = paddle.nn.Linear(in_features=16 * 7 * 7, out_features=10)
        self.flatten = paddle.nn.Flatten(start_axis=1, stop_axis=-1)

    def forward(self, x):
        """
        forward
        """
        x = self.conv1(x)
        x = F.relu(x)
        x = self.max_pool1(x)
        x = self.max_pool2(x)
        # x = paddle.flatten(x, start_axis=1,stop_axis=-1)
        #         x = paddle.flatten(x, start_axis=1,stop_axis=-1)
        x = self.flatten(x)
        x = self.linear1(x)
        return x


def test_jit_script_base():
    """
    test jit when annotation in forward function
    """
    net = LeNet1()
    path = os.path.join(save_path, "jit_script_in_forward")
    func = paddle.jit.to_static(net.forward, [paddle.static.InputSpec(shape=[-1, 1, 224, 224])])
    paddle.jit.save(func, path)
