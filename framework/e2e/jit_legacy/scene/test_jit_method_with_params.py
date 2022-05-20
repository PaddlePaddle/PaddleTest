#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test jit method with params
"""
import os
import shutil
import paddle
from tools import compare


pwd = os.getcwd()
save_path = os.path.join(pwd, "save_path")
if os.path.exists(save_path):
    shutil.rmtree(save_path)
os.mkdir(os.path.join(pwd, "save_path"))
paddle.seed(103)


class LinearNet1(paddle.nn.Layer):
    """
    linear net
    """

    def __init__(self):
        super(LinearNet1, self).__init__()
        self._linear = paddle.nn.Linear(5, 6)

    def forward(self, x):
        """
        forward
        """
        return paddle.tanh(x)

    def another_forward(self, x):
        """
        another forward
        """
        return self._linear(x)


class LinearNet2(paddle.nn.Layer):
    """
    linear net
    """

    def __init__(self):
        super(LinearNet2, self).__init__()
        self._linear = paddle.nn.Linear(5, 6)

    def forward(self, x):
        """
        forward
        """
        return paddle.tanh(x)

    @paddle.jit.to_static
    def another_forward(self, x):
        """
        another forward
        """
        return self._linear(x)


def test_jit_method_with_params1():
    """
    test jit.save func
    """
    layer = LinearNet1()
    inps = paddle.rand([3, 5])

    func = paddle.jit.to_static(layer.another_forward, [paddle.static.InputSpec(shape=[-1, 5])])
    path = os.path.join(save_path, "method_with_params")
    paddle.jit.save(func, path)
    exp = layer.another_forward(inps)
    load_func = paddle.jit.load(path)
    res = load_func(inps)
    compare(res.numpy(), exp.numpy(), delta=1e-10, rtol=1e-10)


def test_jit_method_with_params2():
    """
    test jit.save func
    """
    layer = LinearNet2()

    inps = paddle.rand([3, 5])
    exp = layer.another_forward(inps)

    path = os.path.join(save_path, "method_with_params")
    paddle.jit.save(layer.another_forward, path)
    load_func = paddle.jit.load(path)
    res = load_func(inps)
    compare(res.numpy(), exp.numpy(), delta=1e-10, rtol=1e-10)
