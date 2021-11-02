#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_unique_consecutive
"""


from apibase import APIBase
from apibase import compare
import paddle
import pytest
import numpy as np


class UiqueConsecutive(APIBase):
    """
    test_unique_consecutive
    """

    def hook(self):
        self.types = [np.float32, np.float64, np.int32, np.int64]
        self.enable_backward = False


obj = UiqueConsecutive(paddle.unique_consecutive)


@pytest.mark.api_base_uique_consecutive_vartype
def test_uique_consecutive_base():
    """
    base
    """
    x = np.array([[1, 1, 2, 2, 1, 1], [1, 1, 2, 2, 1, 1]])
    res = np.array([1, 2, 1, 2, 1])
    obj.base(res, x=x)


@pytest.mark.api_base_uique_consecutive_parameters
def test_uique_consecutive0():
    """
    default:
    return_inverse = False
    return_counts = False
    axis = None
    dtype = int64
    """
    x = np.array([[0], [0], [1], [0]])

    res = np.array([0, 1, 0])
    obj.run(res, x=x)


@pytest.mark.api_base_uique_consecutive_parameters
def test_uique_consecutive1():
    """
    return_inverse = True
    return_counts = False
    axis = None
    dtype = int64
    """
    x = np.array([[0], [0], [1], [0]])
    # excepct result
    res = np.array([[0, 1, 0], [0, 0, 1, 2]])
    for place in obj.places:
        if str(place) == "CPUPlace":
            paddle.set_device("cpu")
        else:
            paddle.set_device("gpu:0")

        obj.dtype = np.float32
        paddle.disable_static()
        obj._check_params(res, x=x)
        obj.kwargs["return_inverse"] = True
        dygrapy_forward_res = obj._dygraph_forward()
        length = len(dygrapy_forward_res)
        for i in range(length):
            compare(dygrapy_forward_res[i].numpy(), res[i])


@pytest.mark.api_base_uique_consecutive_parameters
def test_uique_consecutive2():
    """
    return_inverse = True
    return_counts = True
    axis = None
    dtype = int64
    """
    x = np.array([[0], [0], [1], [0], [0], [0]])
    # excepct result
    res = np.array([[0, 1, 0], [0, 0, 1, 2, 2, 2], [2, 1, 3]])
    for place in obj.places:
        if str(place) == "CPUPlace":
            paddle.set_device("cpu")
        else:
            paddle.set_device("gpu:0")

        obj.dtype = np.float32
        paddle.disable_static()
        obj._check_params(res, x=x)
        obj.kwargs["return_inverse"] = True
        obj.kwargs["return_counts"] = True
        dygrapy_forward_res = obj._dygraph_forward()
        length = len(dygrapy_forward_res)
        for i in range(length):
            compare(dygrapy_forward_res[i].numpy(), res[i])


@pytest.mark.api_base_uique_consecutive_parameters
def test_uique_consecutive3():
    """
    return_inverse = True
    return_counts = True
    axis = 0
    dtype = int64
    """
    x = np.array([[0, 2, 2, 0], [0, 2, 2, 1]])
    x = np.repeat(x, 3, 0)
    # excepct result
    res = np.array(([[0, 2, 2, 0], [0, 2, 2, 1]], [0, 0, 0, 1, 1, 1], [3, 3]))
    for place in obj.places:
        if str(place) == "CPUPlace":
            paddle.set_device("cpu")
        else:
            paddle.set_device("gpu:0")

        obj.dtype = np.float32
        paddle.disable_static()
        obj._check_params(res, x=x)
        obj.kwargs["return_inverse"] = True
        obj.kwargs["return_counts"] = True
        obj.kwargs["axis"] = 0
        dygrapy_forward_res = obj._dygraph_forward()
        length = len(dygrapy_forward_res)
        for i in range(length):
            compare(dygrapy_forward_res[i].numpy(), res[i])


@pytest.mark.api_base_uique_consecutive_parameters
def test_uique_consecutive4():
    """
    x = [[1, 2, 2, 1, 1, 0],
        [0, 0, 2, 2, 4, 4],
        [4, 4, 4, 4, 4, 4]]
    axis = 1
    dtype = int64
    """
    x = np.array([[1, 2, 2, 1, 1, 0], [0, 0, 2, 2, 4, 4], [4, 4, 4, 4, 4, 4]])

    res = np.array([[1, 2, 2, 1, 1, 0], [0, 0, 2, 2, 4, 4], [4, 4, 4, 4, 4, 4]])
    obj.run(res, x=x, axis=1)


@pytest.mark.api_base_uique_consecutive_parameters
def test_uique_consecutive5():
    """
    x = [[1, 2, 2, 1, 1, 0],
         [0, 1, 2, 2, 2, 4],
         [4, 4, 4, 4, 4, 4]]
    axis = 1
    dtype = int32
    """
    x = np.array([[1, 2, 2, 1, 1, 0], [0, 1, 2, 2, 2, 4], [4, 4, 4, 4, 4, 4]])

    res = np.array([[1, 2, 2, 1, 0], [0, 1, 2, 2, 4], [4, 4, 4, 4, 4]])
    obj.run(res, x=x, axis=1)


@pytest.mark.api_base_uique_consecutive_parameters
def test_uique_consecutive6():
    """
    dtype = int32
    """
    x = np.array([[0], [0], [1], [0]])

    res = np.array([0, 1, 0])
    obj.run(res, x=x, dtype="int32")


@pytest.mark.api_base_uique_consecutive_parameters
def test_uique_consecutive7():
    """
    static
    return_inverse = True
    return_counts = False
    axis = None
    dtype = int64
    """
    res = np.array([[0, 1, 0], [0, 0, 1, 2]])
    for place in obj.places:
        if str(place) == "CPUPlace":
            paddle.set_device("cpu")
        else:
            paddle.set_device("gpu:0")

        main_program = paddle.static.Program()
        start_program = paddle.static.Program()
        main_program.random_seed = 222
        start_program.random_seed = 222

        feed = {"x": np.array([[0], [0], [1], [0]], dtype="float32")}

        paddle.enable_static()

        with paddle.utils.unique_name.guard():
            with paddle.static.program_guard(main_program=main_program, startup_program=start_program):
                x = paddle.static.data(name="x", shape=[4, 1], dtype="float32")
                # y = paddle.static.data(name='y', shape=[4, ], dtype='float32')
                output = paddle.unique_consecutive(x, return_inverse=True)
                # output = paddle.static.nn.fc(x, 1)
                exe = paddle.static.Executor(place)
                exe.run(start_program)
                static_forward_res = exe.run(main_program, feed=feed, fetch_list=[output], return_numpy=True)

                length = len(static_forward_res)
                for i in range(length):
                    compare(static_forward_res[i], res[i])


@pytest.mark.api_base_uique_consecutive_parameters
def test_uique_consecutive8():
    """
    static
    return_inverse = True
    return_counts = True
    axis = None
    dtype = int64
    """
    x = np.array([[0], [0], [1], [0], [0], [0]])
    res = np.array([[0, 1, 0], [0, 0, 1, 2, 2, 2], [2, 1, 3]])
    for place in obj.places:
        if str(place) == "CPUPlace":
            paddle.set_device("cpu")
        else:
            paddle.set_device("gpu:0")

        main_program = paddle.static.Program()
        start_program = paddle.static.Program()
        main_program.random_seed = 222
        start_program.random_seed = 222

        feed = {
            # "x": np.array([[0], [0], [1], [0]], dtype='float32'),
            "x": np.array([[0], [0], [1], [0], [0], [0]], dtype="float32")
        }

        paddle.enable_static()

        with paddle.utils.unique_name.guard():
            with paddle.static.program_guard(main_program=main_program, startup_program=start_program):
                x = paddle.static.data(name="x", shape=[6, 1], dtype="float32")
                # y = paddle.static.data(name='y', shape=[4, ], dtype='float32')
                output = paddle.unique_consecutive(x, return_inverse=True, return_counts=True)
                exe = paddle.static.Executor(place)
                exe.run(start_program)
                static_forward_res = exe.run(main_program, feed=feed, fetch_list=[output], return_numpy=True)

                length = len(static_forward_res)
                for i in range(length):
                    compare(static_forward_res[i], res[i])


@pytest.mark.api_base_uique_consecutive_parameters
def test_uique_consecutive9():
    """
    static
    return_inverse = True
    return_counts = True
    axis = 0
    dtype = int64
    """
    # excepct result
    res = np.array(([[0, 2, 2, 0], [0, 2, 2, 1]], [0, 0, 0, 1, 1, 1], [3, 3]))
    for place in obj.places:
        if str(place) == "CPUPlace":
            paddle.set_device("cpu")
        else:
            paddle.set_device("gpu:0")

        main_program = paddle.static.Program()
        start_program = paddle.static.Program()
        main_program.random_seed = 222
        start_program.random_seed = 222

        feed = {
            # "x": np.array([[0], [0], [1], [0]], dtype='float32'),
            # "x": np.array([[0], [0], [1], [0], [0], [0]], dtype='float32')
            "x": np.repeat(np.array([[0, 2, 2, 0], [0, 2, 2, 1]], dtype="float32"), 3, 0)
        }

        paddle.enable_static()

        with paddle.utils.unique_name.guard():
            with paddle.static.program_guard(main_program=main_program, startup_program=start_program):
                x = paddle.static.data(name="x", shape=[6, 4], dtype="float32")
                # y = paddle.static.data(name='y', shape=[4, ], dtype='float32')
                output = paddle.unique_consecutive(x, return_inverse=True, return_counts=True, axis=0)
                exe = paddle.static.Executor(place)
                exe.run(start_program)
                static_forward_res = exe.run(main_program, feed=feed, fetch_list=[output], return_numpy=True)

                length = len(static_forward_res)
                for i in range(length):
                    compare(static_forward_res[i], res[i])
