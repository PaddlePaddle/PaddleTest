#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test normal
"""
import paddle
import pytest
import numpy as np
from paddle import fluid

types = [np.float32, np.float64]
if fluid.is_compiled_with_cuda() is True:
    places = [fluid.CPUPlace(), fluid.CUDAPlace(0)]
else:
    places = [fluid.CPUPlace()]


@pytest.mark.api_base_normal_vartype
def test_dygraph1():
    """
    shape=list
    :return:
    """
    for place in places:
        for t in types:
            paddle.disable_static(place)
            res = paddle.normal(shape=[3, 4])
            assert res.shape == [3, 4]
            assert res.numpy().any() <= 1
            assert res.numpy().any() >= -1
            paddle.enable_static()


@pytest.mark.api_base_normal_parameters
def test_dygraph2():
    """
    shape=list, mean=tensor
    :return:
    """
    for place in places:
        for t in types:
            paddle.disable_static(place)
            mean_tensor = paddle.to_tensor(np.array([1, 2, 3])).astype(t)
            res = paddle.normal(shape=[3, 4], mean=mean_tensor)
            assert res.shape == [3]
            paddle.enable_static()


@pytest.mark.api_base_normal_parameters
def test_dygraph3():
    """
    shape=list, mean=tensor
    :return:
    """
    for place in places:
        for t in types:
            paddle.disable_static(place)
            mean_tensor = paddle.to_tensor(np.array([1, 2, 3])).astype(t)
            res = paddle.normal(shape=[3, 4], mean=mean_tensor)
            assert res.shape == [3]
            paddle.enable_static()


@pytest.mark.api_base_normal_parameters
def test_dygraph4():
    """
    shape=tuple, mean=tensor, std=tensor
    :return:
    """
    for place in places:
        for t in types:
            paddle.disable_static(place)
            mean_tensor = paddle.to_tensor(np.array([1, 2, 3])).astype(t)
            std_tensor = paddle.to_tensor(np.array([1, 2, 3])).astype(t)
            res = paddle.normal(shape=(3, 4), mean=mean_tensor, std=std_tensor)
            assert res.shape == [3]
            paddle.enable_static()


@pytest.mark.api_base_normal_parameters
def test_dygraph5():
    """
    shape=tensor, mean=tensor, std=tensor
    :return:
    """
    for place in places:
        for t in types:
            paddle.disable_static(place)
            mean_tensor = paddle.to_tensor(np.array([1, 2, 3])).astype(t)
            std_tensor = paddle.to_tensor(np.array([1, 2, 3])).astype(t)
            shape_tensor = paddle.to_tensor(np.array([3, 4]).astype(t))
            res = paddle.normal(shape=shape_tensor, mean=mean_tensor, std=std_tensor)
            assert res.shape == [3]
            paddle.enable_static()


@pytest.mark.api_base_normal_vartype
def test_static1():
    """
    shape=list
    :return:
    """
    for place in places:
        for t in types:
            main_program = fluid.default_main_program()
            startup_program = fluid.default_startup_program()
            # shape = [3, 4]
            with fluid.unique_name.guard():
                with fluid.program_guard(main_program=main_program, startup_program=startup_program):
                    exe = fluid.Executor(place)
                    exe.run(startup_program)
                    output = paddle.normal(shape=[3, 4])
                    res = exe.run(main_program, fetch_list=[output])
                    assert res[0].shape == (3, 4)


@pytest.mark.api_base_normal_parameters
def test_static2():
    """
    shape=list, mean=tensor
    :return:
    """
    for place in places:
        for t in types:
            main_program = fluid.default_main_program()
            startup_program = fluid.default_startup_program()
            # shape = [3, 4]
            mean = np.array([1, 2, 3]).astype("float32")
            feed = {"mean": mean}
            with fluid.unique_name.guard():
                with fluid.program_guard(main_program=main_program, startup_program=startup_program):
                    mean1 = fluid.data(name="mean", shape=[-1], dtype="float32")
                    exe = fluid.Executor(place)
                    exe.run(startup_program)
                    output = paddle.normal(shape=[3, 4], mean=mean1)
                    res = exe.run(main_program, feed=feed, fetch_list=[output])
                    assert res[0].shape == (3,)


@pytest.mark.api_base_normal_parameters
def test_static3():
    """
    shape=list, mean=tensor, std=tensor
    :return:
    """
    for place in places:
        for t in types:
            main_program = fluid.default_main_program()
            startup_program = fluid.default_startup_program()
            # shape = [3, 4]
            mean = np.array([1, 2, 3]).astype("float32")
            std = np.array([1, 2, 3]).astype("float32")
            feed = {"mean": mean, "std": std}
            with fluid.unique_name.guard():
                with fluid.program_guard(main_program=main_program, startup_program=startup_program):
                    mean1 = fluid.data(name="mean", shape=[-1], dtype="float32")
                    std1 = fluid.data(name="std", shape=[-1], dtype="float32")
                    exe = fluid.Executor(place)
                    exe.run(startup_program)
                    output = paddle.normal(shape=[3, 4], mean=mean1, std=std1)
                    res = exe.run(main_program, feed=feed, fetch_list=[output])
                    assert res[0].shape == (3,)
