#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_put_along_axis
"""

from apibase import randtool
import paddle
import pytest
import numpy as np


class PutAlongAxis(object):
    """
    calculate put_along_axis api
    """

    def __init__(self, indices, values, axis, types, reduce="assign"):
        """
        init
        """
        self.indices = indices
        self.values = values
        self.axis = axis
        self.reduce = reduce
        self.types = types

        if paddle.device.is_compiled_with_cuda() is True:
            self.places = ["cpu", "gpu:0"]
        else:
            self.places = ["cpu"]

    def cal_dynamic(self, device, dtype):
        """
        dynamic calculate
        """
        paddle.set_device(device)
        paddle.disable_static()
        arr = paddle.to_tensor(self.arr, dtype=dtype)
        indices = paddle.to_tensor(self.indices)
        values = paddle.to_tensor(self.values, dtype=dtype)
        return paddle.Tensor.put_along_axis_(arr, indices, values, self.axis, self.reduce)

    def cal_static(self, device):
        """
        static_calculate
        """
        paddle.enable_static()
        paddle.set_device(device)
        main_program, startup_program = paddle.static.Program(), paddle.static.Program()
        with paddle.utils.unique_name.guard():
            with paddle.static.program_guard(main_program=main_program, startup_program=startup_program):
                data0 = paddle.static.data(name="s0", shape=self.arr.shape, dtype=self.arr.dtype)
                data1 = paddle.static.data(name="s1", shape=self.indices.shape, dtype=self.indices.dtype)
                data2 = paddle.static.data(name="s2", shape=self.values.shape, dtype=self.values.dtype)
                feed = {"s0": self.arr, "s1": self.indices, "s2": self.values}
                out = paddle.Tensor.put_along_axis_(
                    data0, indices=data1, values=data2, axis=self.axis, reduce=self.reduce
                )
                exe = paddle.static.Executor(device)
                exe.run(startup_program)
                static_res = exe.run(main_program, feed=feed, fetch_list=[out])
        paddle.disable_static()
        return static_res

    def run(self, arr):
        """
        run api
        """
        self.arr = arr
        res_record = None
        for device in self.places:
            for dtype in self.types:
                self.arr = self.arr.astype(dtype)
                self.values = self.values.astype(dtype)
                d_res = self.cal_dynamic(device, dtype)
                s_res = self.cal_static(device)
                assert np.allclose(d_res.numpy(), s_res[0])
                res_record = d_res.numpy()
        return res_record


@pytest.mark.api_base_put_along_axis_vartype
def test_put_along_axis_base():
    """
    base
    """
    data = randtool("float", 0, 1, (4,))
    types = ["float32", "float64"]
    ind = np.array([0])
    value = np.array([100])
    axis = 0
    res = data.copy()
    res[0] = 100
    obj = PutAlongAxis(ind, value, axis, types)
    r = obj.run(data)
    assert np.allclose(r, res)


@pytest.mark.api_base_put_along_axis_parameters
def test_put_along_axis0():
    """
    arr: 2d-tensor
    """
    data = randtool("float", 0, 1, (4, 4))
    types = ["float64"]
    ind = np.array([[0]])
    value = np.array([100])
    axis = 0
    res = data.copy()
    res[0] = 100
    obj = PutAlongAxis(ind, value, axis, types)
    r = obj.run(data)
    assert np.allclose(r, res)


@pytest.mark.api_base_put_along_axis_parameters
def test_put_along_axis1():
    """
    arr: 3d-tensor
    """
    data = randtool("float", 0, 1, (4, 4, 4))
    types = ["float64"]
    ind = np.array([[[0]]])
    value = np.array([100])
    axis = 0
    res = data.copy()
    res[0] = 100
    obj = PutAlongAxis(ind, value, axis, types)
    r = obj.run(data)
    assert np.allclose(r, res)


@pytest.mark.api_base_put_along_axis_parameters
def test_put_along_axis2():
    """
    arr: 4d-tensor
    """
    data = randtool("float", 0, 1, (4, 4, 4, 2))
    types = ["float64"]
    ind = np.array([[[[0]]]])
    value = np.array([100])
    axis = 0
    res = data.copy()
    res[0] = 100
    obj = PutAlongAxis(ind, value, axis, types)
    r = obj.run(data)
    assert np.allclose(r, res)


@pytest.mark.api_base_put_along_axis_parameters
def test_put_along_axis3():
    """
    arr: 2d-tensor
    indices = [[1]]
    """
    data = randtool("float", 0, 1, (4, 2))
    types = ["float64"]
    ind = np.array([[1]])
    value = np.array([100])
    axis = 0
    res = data.copy()
    res[1] = 100
    obj = PutAlongAxis(ind, value, axis, types)
    r = obj.run(data)
    assert np.allclose(r, res)


@pytest.mark.api_base_put_along_axis_parameters
def test_put_along_axis4():
    """
    arr: 2d-tensor
    value < 0
    """
    data = randtool("float", 0, 5, (2, 2))
    types = ["float64"]
    ind = np.array([[1]])
    value = np.array([-100])
    axis = 0
    res = data.copy()
    res[1] = -100
    obj = PutAlongAxis(ind, value, axis, types)
    r = obj.run(data)
    assert np.allclose(r, res)


@pytest.mark.api_base_put_along_axis_parameters
def test_put_along_axis5():
    """
    arr: 2d-tensor
    value < 0
    axis = 1
    """
    data = randtool("float", 0, 5, (2, 2))
    types = ["float64"]
    ind = np.array([[1]])
    value = np.array([-100])
    axis = 1
    res = data.copy()
    res[:, 1] = -100
    obj = PutAlongAxis(ind, value, axis, types)
    r = obj.run(data)
    assert np.allclose(r, res)


@pytest.mark.api_base_put_along_axis_parameters
def test_put_along_axis6():
    """
    arr: 2d-tensor
    value < 0
    axis = 1
    reduce = add
    """
    data = randtool("float", 0, 5, (2, 2))
    types = ["float64"]
    ind = np.array([[1]])
    value = np.array([-100])
    axis = 1
    res = data.copy()
    res[:, 1] += -100
    obj = PutAlongAxis(ind, value, axis, types, reduce="add")
    r = obj.run(data)
    assert np.allclose(r, res)


@pytest.mark.api_base_put_along_axis_parameters
def test_put_along_axis7():
    """
    arr: 2d-tensor
    value < 0
    axis = 1
    reduce = multiple
    """
    data = randtool("float", 0, 5, (2, 2))
    types = ["float64"]
    ind = np.array([[1]])
    value = np.array([-100])
    axis = 1
    res = data.copy()
    res[:, 1] *= -100
    obj = PutAlongAxis(ind, value, axis, types, reduce="mul")
    r = obj.run(data)
    assert np.allclose(r, res)
