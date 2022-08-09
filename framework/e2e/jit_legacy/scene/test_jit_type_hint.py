#!/bin/env python3
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test jit type hint
"""
import os
from typing import Tuple, List, Dict, TypeVar
import numpy as np
import paddle
from tools import compare

np.random.seed(102)
paddle.seed(123)
pwd = os.getcwd()
save_path = os.path.join(pwd, "save_path")
if not os.path.exists(save_path):
    os.mkdir(save_path)


class BaseLayer(paddle.nn.Layer):
    """
    base layer
    """

    def __init__(self, in_channels, out_channels):
        super(BaseLayer, self).__init__()
        self._conv = paddle.nn.Conv2D(in_channels, out_channels, 3)
        self._bn = paddle.nn.BatchNorm2D(out_channels)

    def build(self, x):
        """
        build
        """
        out1 = self._conv(x)
        out1 = self._bn(out1)
        exp = paddle.mean(out1)
        return out1, exp


class LinearNetWithTuple2(BaseLayer):
    """
    linear net with tuple
    """

    def __init__(self, in_size, out_size):
        super(LinearNetWithTuple2, self).__init__(in_size, out_size)

    def forward(self, x) -> Tuple[paddle.Tensor, np.array]:
        """
        forward
        """
        out1, exp = self.build(x)
        return (exp, np.ones([4, 16]))


class LinearNetWithDict(BaseLayer):
    """
    linear net with dict
    """

    def __init__(self, in_size, out_size):
        super(LinearNetWithDict, self).__init__(in_size, out_size)

    def forward(self, x) -> Dict[str, paddle.Tensor]:
        """
        forward
        """
        out1, exp = self.build(x)
        return {"out": exp}


def test_jit_tuple_hint():
    """
    test jit type hint
    """
    path = os.path.join(save_path, "jit_tuple_hint")

    layer_dy = LinearNetWithTuple2(in_size=3, out_size=10)
    layer_dy.eval()
    layer_st = paddle.jit.to_static(layer_dy)
    x = paddle.rand([2, 3, 224, 224], "float32")
    exp = layer_st(x)
    paddle.jit.save(layer_st, path)
    load_func = paddle.jit.load(path)
    result = load_func(x)
    compare(result.numpy(), exp[0].numpy(), delta=1e-10, rtol=1e-10)


def test_jit_dict_hint():
    """
    test jit type hint
    """
    path = os.path.join(save_path, "jit_dict_hint")

    layer_dy = LinearNetWithDict(in_size=3, out_size=10)
    layer_dy.eval()
    layer_st = paddle.jit.to_static(layer_dy)
    x = paddle.rand([2, 3, 224, 224], "float32")
    exp = layer_st(x)
    paddle.jit.save(layer_st, path)
    load_func = paddle.jit.load(path)
    result = load_func(x)
    compare(result.numpy(), exp["out"].numpy(), delta=1e-10, rtol=1e-10)
