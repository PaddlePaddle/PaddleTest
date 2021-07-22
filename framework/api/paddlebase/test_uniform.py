#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test uniform
"""
import paddle
import pytest
import numpy as np
import paddle.fluid as fluid


types = [np.float32, np.float64]
if fluid.is_compiled_with_cuda() is True:
    places = [fluid.CPUPlace(), fluid.CUDAPlace(0)]
else:
    places = [fluid.CPUPlace()]


@pytest.mark.api_base_uniform_vartype
def test_dygraph1():
    """
    shape=list
    :return:
    """
    for place in places:
        for t in types:
            paddle.disable_static(place)
            res = paddle.uniform(shape=[3, 4], dtype=t)
            assert res.shape == [3, 4]
            paddle.enable_static()


@pytest.mark.api_base_uniform_parameters
def test_dygraph2():
    """
    shape=tuple
    :return:
    """
    for place in places:
        for t in types:
            paddle.disable_static(place)
            res = paddle.uniform(shape=(3, 4), dtype=t)
            assert res.shape == [3, 4]
            paddle.enable_static()


@pytest.mark.api_base_uniform_parameters
def test_dygraph3():
    """
    shape=list[tensor]
    :return:
    """
    for place in places:
        for t in types:
            paddle.disable_static(place)
            # dim_1 = paddle.fill_constant([1], "int64", 2) 新版本2.0rc不存在此api
            # dim_2 = paddle.fill_constant([1], "int64", 3)
            dim_1 = paddle.to_tensor([2], "int64")
            dim_2 = paddle.to_tensor([3], "int64")
            res = paddle.uniform(shape=(dim_1, dim_2), dtype=t)
            assert res.shape == [2, 3]
            paddle.enable_static()


@pytest.mark.api_base_uniform_parameters
def test_dygraph4():
    """
    shape=[tensor]
    :return:
    """
    for place in places:
        for t in types:
            paddle.disable_static(place)
            shape = np.array([2, 3]).astype(t)
            shape_tensor = paddle.to_tensor(shape)
            res = paddle.uniform(shape=shape_tensor, dtype=t)
            assert res.shape == [2, 3]
            paddle.enable_static()


@pytest.mark.api_base_uniform_parameters
def test_dygraph5():
    """
    shape=list, min=-1.0, max=1.0
    :return:
    """
    for place in places:
        for t in types:
            paddle.disable_static(place)
            res = paddle.uniform(shape=[3, 4], dtype=t, min=-1.0, max=1.0)
            assert res.shape == [3, 4]
            paddle.enable_static()


@pytest.mark.api_base_uniform_parameters
def test_dygraph6():
    """
    shape=list, min=-1, max=1
    :return:
    """
    for place in places:
        for t in types:
            paddle.disable_static(place)
            res = paddle.uniform(shape=[3, 4], dtype=t, min=-1, max=1)
            assert res.shape == [3, 4]
            assert res.numpy().any() <= 1
            assert res.numpy().any() >= -1
            paddle.enable_static()


@pytest.mark.api_base_uniform_parameters
def test_dygraph7():
    """
    shape=list, min=-1, max=1
    :return:
    """
    for place in places:
        for t in types:
            paddle.disable_static(place)
            res1 = paddle.uniform(shape=[3, 4], dtype=t, min=-1, max=1, seed=2)
            res2 = paddle.uniform(shape=[3, 4], dtype=t, min=-1, max=1, seed=2)
            assert res1.shape == res2.shape == [3, 4]
            assert res1.numpy().any() == res2.numpy().any()
            paddle.enable_static()


@pytest.mark.api_base_uniform_vartype
def test_static1():
    """
    shape=list
    :return:
    """
    for place in places:
        for t in types:
            main_program = fluid.default_main_program()
            startup_program = fluid.default_startup_program()
            shape = [3, 4]
            with fluid.unique_name.guard():
                with fluid.program_guard(main_program=main_program, startup_program=startup_program):
                    exe = fluid.Executor(place)
                    exe.run(startup_program)
                    output = paddle.uniform(shape=shape)
                    res = exe.run(main_program, fetch_list=[output])
                    assert res[0].shape == (3, 4)


@pytest.mark.api_base_uniform_parameters
def test_static2():
    """
    shape=tuple
    :return:
    """
    for place in places:
        for t in types:
            main_program = fluid.default_main_program()
            startup_program = fluid.default_startup_program()
            shape = (3, 4)
            with fluid.unique_name.guard():
                with fluid.program_guard(main_program=main_program, startup_program=startup_program):
                    exe = fluid.Executor(place)
                    exe.run(startup_program)
                    output = paddle.uniform(shape=shape)
                    res = exe.run(main_program, fetch_list=[output])
                    assert res[0].shape == (3, 4)


@pytest.mark.api_base_uniform_parameters
def test_static3():
    """
    shape=tensor, tensor, shape_tensor=int32
    :return:
    """
    for place in places:
        for t in types:
            main_program = fluid.default_main_program()
            startup_program = fluid.default_startup_program()
            shape = np.array([3, 4]).astype("int32")
            feed = {"shape": shape}
            with fluid.unique_name.guard():
                with fluid.program_guard(main_program=main_program, startup_program=startup_program):
                    shape = fluid.data(name="shape", shape=[2], dtype="int32")
                    exe = fluid.Executor(place)
                    exe.run(startup_program)
                    output = paddle.uniform(shape=shape)
                    res = exe.run(main_program, feed=feed, fetch_list=[output])
                    assert res[0].shape == (3, 4)


@pytest.mark.api_base_uniform_parameters
def test_static4():
    """
    shape=tensor, tensor, shape_type=int64
    :return:
    """
    for place in places:
        for t in types:
            main_program = fluid.default_main_program()
            startup_program = fluid.default_startup_program()
            shape = np.array([3, 4]).astype("int32")
            feed = {"shape": shape}
            with fluid.unique_name.guard():
                with fluid.program_guard(main_program=main_program, startup_program=startup_program):
                    shape = fluid.data(name="shape", shape=[2], dtype="int32")
                    exe = fluid.Executor(place)
                    exe.run(startup_program)
                    output = paddle.uniform(shape=shape)
                    res = exe.run(main_program, feed=feed, fetch_list=[output])
                    assert res[0].shape == (3, 4)


@pytest.mark.api_base_uniform_parameters
def test_static5():
    """
    shape=tensor, tensor, shape_type=int64,
    :return:
    """
    for place in places:
        main_program = fluid.default_main_program()
        startup_program = fluid.default_startup_program()
        shape = np.array([3, 4]).astype("int32")
        feed = {"shape": shape}
        with fluid.unique_name.guard():
            with fluid.program_guard(main_program=main_program, startup_program=startup_program):
                shape = fluid.data(name="shape", shape=[2], dtype="int32")
                exe = fluid.Executor(place)
                exe.run(startup_program)
                output = paddle.uniform(shape=shape, dtype="float32", min=-1, max=1)
                res = exe.run(main_program, feed=feed, fetch_list=[output])
                assert res[0].shape == (3, 4)
