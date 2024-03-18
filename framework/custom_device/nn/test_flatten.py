#!/usr/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python

"""
test_flatten
"""

from apibase import APIBase
import paddle
import pytest
import numpy as np


class TestFaltten(APIBase):
    """
    test flatten
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32, np.float64, np.int32, np.int64]
        # self.debug = True
        # enable check grad
        # self.enable_backward = False


obj = TestFaltten(paddle.nn.Flatten)


@pytest.mark.api_nn_flatten_vartype
def test_flatten_base():
    """
    x=np.ones([2, 2, 3])
    """
    x = np.ones(shape=[2, 4])
    res = np.ones(shape=[2, 4])
    obj.base(res=res, data=x)


@pytest.mark.api_nn_flatten_parameters
def test_flatten1():
    """
    x=np.ones([2, 2, 3]),start_axis=0,stop_axis=-1,default
    """
    x = np.ones(shape=[2, 2, 3])
    res = np.ones(shape=[12])
    start_axis = 0
    stop_axis = -1
    obj.run(res=res, data=x, start_axis=start_axis, stop_axis=stop_axis)


@pytest.mark.api_nn_flatten_parameters
def test_flatten2():
    """
    x=np.ones([2, 2, 3]),stop_axis=-1,default
    """
    x = np.ones(shape=[2, 2, 3])
    res = np.ones(shape=[2, 6])
    stop_axis = -1
    obj.run(res=res, data=x, stop_axis=stop_axis)


@pytest.mark.api_nn_flatten_parameters
def test_flatten3():
    """
    x=np.ones([2, 2, 3]),start_axis=-1,default
    """
    x = np.ones(shape=[2, 2, 3])
    res = np.ones(shape=[2, 2, 3])
    start_axis = -1
    obj.run(res=res, data=x, start_axis=start_axis)


@pytest.mark.api_nn_flatten_parameters
def test_flatten4():
    """
    x=np.ones([1, 2, 3]),start_axis=1,stop_axis=2
    """
    x = np.ones(shape=[1, 2, 3])
    start_axis = 1
    stop_axis = 2
    res = np.ones(shape=[1, 6])
    obj.run(res=res, data=x, start_axis=start_axis, stop_axis=stop_axis)


@pytest.mark.api_nn_flatten_parameters
def test_flatten5():
    """
    x=np.ones([1, 2, 3]),start_axis=stop_axis
    """
    x = np.ones(shape=[1, 2, 3])
    start_axis = -3
    stop_axis = -3
    res = x
    obj.run(res=res, data=x, start_axis=start_axis, stop_axis=stop_axis)


@pytest.mark.api_nn_flatten_parameters
def test_flatten6():
    """
    x=np.zeros([2, 2, 3]),start_axis=-3,stop_axis=1
    """
    x = np.zeros(shape=[2, 2, 3])
    start_axis = -3
    stop_axis = 1
    res = np.zeros(shape=[4, 3])
    obj.run(res=res, data=x, start_axis=start_axis, stop_axis=stop_axis)


@pytest.mark.api_nn_flatten_parameters
def test_flatten7():
    """
    x=np.ones([2, 3, 4]),start_axis=-2,stop_axis=1
    """
    x = np.ones(shape=[2, 3, 4])
    start_axis = -2
    stop_axis = 1
    res = x
    obj.run(res=res, data=x, start_axis=start_axis, stop_axis=stop_axis)


@pytest.mark.api_nn_flatten_parameters
def test_flatten8():
    """
    x=np.ones([1, 2, 3]),start_axis=0,stop_axis=1
    """
    x = np.ones(shape=[1, 2, 3])
    start_axis = 0
    stop_axis = 1
    res = np.ones(shape=[2, 3])
    obj.run(res=res, data=x, start_axis=start_axis, stop_axis=stop_axis)


@pytest.mark.api_nn_flatten_parameters
def test_flatten9():
    """
    x=np.zeros([2, 2, 3]),start_axis=0,stop_axis=2
    """
    x = np.zeros(shape=[2, 2, 3])
    start_axis = 0
    stop_axis = 2
    res = np.zeros(shape=[12])
    obj.run(res=res, data=x, start_axis=start_axis, stop_axis=stop_axis)


@pytest.mark.api_nn_flatten_parameters
def test_flatten10():
    """
    x=np.zeros([3, 100, 100])
    """
    x = np.zeros(shape=[3, 100, 100])
    res = np.zeros(shape=[3, 100 * 100])
    obj.run(res=res, data=x)


@pytest.mark.api_nn_flatten_parameters
def test_flatten11():
    """
    x=np.zeros([2, 3, 4, 5]),start_axis=-3,stop_axis=-1
    """
    x = np.zeros(shape=[2, 3, 4, 5])
    start_axis = -3
    stop_axis = -1
    res = np.zeros(shape=[2, 60])
    obj.run(res=res, data=x, start_axis=start_axis, stop_axis=stop_axis)


@pytest.mark.api_nn_flatten_parameters
def test_flatten12():
    """
    x=np.ones([2, 3, 4, 5]),start_axis=-2,stop_axis=-1
    """
    x = np.ones(shape=[2, 3, 4, 5])
    start_axis = -2
    stop_axis = -1
    res = np.ones(shape=[2, 3, 20])
    obj.run(res=res, data=x, start_axis=start_axis, stop_axis=stop_axis)
