#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test jit func without param
"""
import os
import shutil
import paddle
from paddle.static import InputSpec
from tools import compare, randtool, delete_all


pwd = os.getcwd()
save_path = os.path.join(pwd, "save_path")
if os.path.exists(save_path):
    shutil.rmtree(save_path)
os.mkdir(os.path.join(pwd, "save_path"))

paddle.seed(103)


def test_function1():
    """
    paddle.jit.save(function)
    @paddle.jit.to_static
    function: paddle.tanh
    """

    @paddle.jit.to_static
    def fun(inputs):
        """
        paddle.tanh
        """
        return paddle.tanh(inputs)

    path = os.path.join(save_path, "paddle_tanh")
    inps = paddle.rand([3, 6, 2, 1, 2, 2, 2, 3]) * 10
    expect = fun(inps)

    paddle.jit.save(fun, path)
    load_func = paddle.jit.load(path)

    result = load_func(inps)
    compare(result.numpy(), expect, delta=1e-10, rtol=1e-10)
    delete_all(save_path)


def test_function1_1():
    """
    paddle.jit.save(function)
    function: paddle.tanh
    """

    def fun(inputs):
        """
        paddle.tanh
        """
        return paddle.tanh(inputs)

    path = os.path.join(save_path, "paddle_tanh")
    inps = paddle.rand([3, 6])
    expect = fun(inps)

    paddle.jit.save(fun, path, input_spec=[InputSpec(shape=[None, 6], dtype="float32", name="x")])
    load_func = paddle.jit.load(path)

    result = load_func(inps)
    compare(result.numpy(), expect, delta=1e-10, rtol=1e-10)
    delete_all(save_path)


def test_function1_2():
    """
    paddle.jit.save(function)
    @paddle.jit.to_static
    function: paddle.tanh
    """

    @paddle.jit.to_static
    def fun(inputs):
        """
        paddle.tanh
        """
        return paddle.tanh(inputs)

    path = os.path.join(save_path, "paddle_tanh")
    inps = paddle.rand([3, 6, 2, 1, 2, 2, 2, 3]) * 10

    paddle.jit.save(fun, path, input_spec=[InputSpec(shape=[None, 6], dtype="float32", name="x")])

    expect = fun(inps)
    load_func = paddle.jit.load(path)

    result = load_func(inps)
    compare(result.numpy(), expect, delta=1e-10, rtol=1e-10)
    delete_all(save_path)


def test_function2():
    """
    paddle.jit.save(function)
    @paddle.jit.to_static(input_spec)
    function: paddle.nn.functional.relu
    """

    @paddle.jit.to_static(input_spec=[InputSpec(shape=[None, 6], dtype="float32", name="x")])
    def fun3(inputs):
        """
        paddle.nn.functional.relu
        """
        return paddle.nn.functional.relu(inputs)

    path = os.path.join(save_path, "paddle_nn_functional_relu")
    inps = paddle.rand([3, 3, 5, 1, 1, 2, 6, 8, 10]) * 10
    expect = fun3(inps)

    paddle.jit.save(fun3, path)
    load_func = paddle.jit.load(path)
    result = load_func(inps)
    compare(result.numpy(), expect, delta=1e-10, rtol=1e-10)
    delete_all(save_path)


def test_function2_1():
    """
    paddle.jit.save(function)
    @paddle.jit.to_static(input_spec)
    function: paddle.nn.functional.relu
    """

    @paddle.jit.to_static(input_spec=[InputSpec(shape=[None, 6], dtype="float32", name="x")])
    def fun(inputs):
        """
        paddle.nn.functional.relu
        """
        return paddle.nn.functional.relu(inputs)

    path = os.path.join(save_path, "paddle_nn_functional_relu")
    inps = paddle.rand([3, 3, 5, 1, 1, 2, 6, 8, 10]) * 10

    paddle.jit.save(fun, path)
    expect = fun(inps)
    load_func = paddle.jit.load(path)
    result = load_func(inps)
    compare(result.numpy(), expect, delta=1e-10, rtol=1e-10)
    delete_all(save_path)


def test_function3():
    """
    paddle.jit.save(function)
    @paddle.jit.to_static(input_spec)
    function: paddle.nn.functional.relu6
    """

    @paddle.jit.to_static(input_spec=[InputSpec(shape=[None, 3, 5, 1, 1, 2, 6, 8, 10], dtype="float32", name="x")])
    def fun(inputs):
        """
        paddle.nn.functional.relu6
        """
        return paddle.nn.functional.relu6(inputs)

    path = os.path.join(save_path, "paddle_nn_functional_relu")
    inps = paddle.rand([3, 3, 5, 1, 1, 2, 6, 8, 10]) * 10
    expect = fun(inps)

    paddle.jit.save(fun, path)
    load_func = paddle.jit.load(path)
    result = load_func(inps)
    compare(result.numpy(), expect, delta=1e-10, rtol=1e-10)
    delete_all(save_path)


def test_function4():
    """
    paddle.jit.save(function)
    @paddle.jit.to_static(input_spec)
    function: paddle.nn.functional.avg_pool2d
    """

    @paddle.jit.to_static(input_spec=[InputSpec(shape=[None, 3, 224, 224], dtype="float32", name="x")])
    def fun(inputs):
        """
        paddle.nn.functional.avg_pool2d
        """
        return paddle.nn.functional.avg_pool2d(inputs, kernel_size=3)

    path = os.path.join(save_path, "paddle_nn_functional_avg_pool2d")
    inps = paddle.rand([3, 3, 224, 224])
    expect = fun(inps)

    paddle.jit.save(fun, path)
    load_func = paddle.jit.load(path)
    result = load_func(inps)
    compare(result.numpy(), expect, delta=1e-10, rtol=1e-10)
    delete_all(save_path)


def test_function5():
    """
    paddle.jit.save(function)
    @paddle.jit.to_static(input_spec)
    function: paddle.nn.functional.adaptive_avg_pool2d
    """

    @paddle.jit.to_static(input_spec=[InputSpec(shape=[None, 3, 224, 224], dtype="float32", name="x")])
    def fun(inputs):
        """
        paddle.nn.functional.adaptive_avg_pool2d
        """
        return paddle.nn.functional.adaptive_avg_pool2d(inputs, output_size=112)

    path = os.path.join(save_path, "paddle_nn_functional_avg_pool2d")
    inps = paddle.rand([3, 3, 224, 224])
    expect = fun(inps)

    paddle.jit.save(fun, path)
    load_func = paddle.jit.load(path)
    result = load_func(inps)
    compare(result.numpy(), expect, delta=1e-10, rtol=1e-10)
    delete_all(save_path)


def test_function6():
    """
    paddle.jit.save(function)
    @paddle.jit.to_static
    function: paddle.log
    """

    @paddle.jit.to_static
    def fun(inputs):
        """
        paddle.log
        """
        return paddle.log(inputs)

    path = os.path.join(save_path, "paddle_log")
    inps = paddle.rand([3, 6])
    expect = fun(inps)

    paddle.jit.save(fun, path)
    load_func = paddle.jit.load(path)

    result = load_func(inps)
    compare(result.numpy(), expect, delta=1e-10, rtol=1e-10)
    delete_all(save_path)


def test_delete_all():
    """
    delete save path
    """
    shutil.rmtree(save_path)


# def test_function7():
#     """
#     paddle.jit.save(function)
#     @paddle.jit.to_static
#     function: paddle.nn.functional.cross_entropy
#     """
#     @paddle.jit.to_static(input_spec=[
#         InputSpec(shape=[None, 3, 224, 224], dtype='float32', name='x'),
#         InputSpec(shape=[None, 1], dtype='float32', name='y'), ])
#     def fun5(inputs, labels):
#         return paddle.nn.functional.cross_entropy(inputs, labels)
#
#     path = os.path.join(save_path, 'paddle_nn_functional_relu')
#     inps = paddle.rand([3, 3, 224, 224])
#     label = paddle.to_tensor([0, 1, 2, 1])
#     expect = fun5(inps, label)
#
#     paddle.jit.save(fun5, path)
#     load_func = paddle.jit.load(path)
#
#     result = load_func(inps)
#     compare(result.numpy(), expect, delta=1e-10, rtol=1e-10)
#     delete_all(save_path)
