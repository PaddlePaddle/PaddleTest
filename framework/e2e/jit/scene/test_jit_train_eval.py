#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test Conv2D_Transpose
"""
import os
import paddle
import numpy as np
from tools import compare
from tools import randtool


np.random.seed(102)
paddle.seed(123)
pwd = os.getcwd()
save_path = os.path.join(pwd, "save_path")
if not os.path.exists(save_path):
    os.mkdir(save_path)


@paddle.jit.to_static
def func(inputs, inputs_):
    """
    paddle.abs
    """
    out = paddle.nn.functional.conv2d_transpose(
        inputs,
        inputs_,
        bias=None,
        stride=1,
        padding=0,
        output_padding=0,
        groups=1,
        dilation=1,
        data_format="NCHW",
        output_size=None,
    )

    return out


def test_jit_train():
    """
    test jit train
    """
    path = os.path.join(save_path, "jit_train")

    func.train()
    func_st = paddle.jit.to_static(func)
    inputs = randtool("float", -2, 2, shape=[2, 3, 8, 8])
    inputs_ = randtool("float", -2, 2, shape=[3, 6, 3, 3])
    exp = func_st(inputs, inputs_)
    paddle.jit.save(func_st, path)
    load_func = paddle.jit.load(path)
    result = load_func(inputs, inputs_)
    compare(result.numpy(), exp.numpy(), delta=1e-10, rtol=1e-10)


def test_jit_eval():
    """
    test jit eval
    """
    path = os.path.join(save_path, "jit_eval")

    func.eval()
    func_st = paddle.jit.to_static(func)
    inputs = randtool("float", -2, 2, shape=[2, 3, 8, 8])
    inputs_ = randtool("float", -2, 2, shape=[3, 6, 3, 3])
    exp = func_st(inputs, inputs_)
    paddle.jit.save(func_st, path)
    load_func = paddle.jit.load(path)
    result = load_func(inputs, inputs_)
    compare(result.numpy(), exp.numpy(), delta=1e-10, rtol=1e-10)
