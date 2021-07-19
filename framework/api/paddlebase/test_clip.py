#!/usr/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python

"""
test clip
"""
from apibase import APIBase
from apibase import randtool
from apibase import compare
import paddle
import pytest
import numpy as np


class TestClip(APIBase):
    """
    test clip
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32, np.float64]
        # self.debug = True
        # enable check grad
        self.enable_backward = False


obj = TestClip(paddle.clip)


@pytest.mark.api_base_clip_vartype
def test_clip_base():
    """
    base
    """
    x = randtool("float", -10, 10, [3, 3])
    min = -5.0
    max = 5.0
    res = np.clip(a=x, a_min=min, a_max=max)
    obj.base(res=res, x=x, min=min, max=max)


@pytest.mark.api_base_clip_parameters
def test_clip1():
    """
    min < x,max = 5
    """
    x = randtool("float", 1, 10, [3, 3])
    min = 0.0
    max = 5
    res = np.clip(a=x, a_min=min, a_max=max)
    obj.run(res=res, x=x, min=min, max=max)


@pytest.mark.api_base_clip_parameters
def test_clip2():
    """
    min =x,max=None,x = np.ones([3, 3])
    """
    x = np.ones([3, 3])
    min = 1.0
    max = None
    res = np.clip(a=x, a_min=min, a_max=max)
    obj.run(res=res, x=x, min=min, max=max)


@pytest.mark.api_base_clip_parameters
def test_clip3():
    """
    min >x,max=None
    """
    x = randtool("float", -1, 10, [3, 3])
    min = 5
    max = None
    res = np.clip(a=x, a_min=min, a_max=max)
    obj.run(res=res, x=x, min=min, max=max)


@pytest.mark.api_base_clip_parameters
def test_clip4():
    """
    min =x=max
    """
    x = np.array([-10, 3, 0])
    min = 2.0
    max = 2.0
    res = np.clip(a=x, a_min=min, a_max=max)
    obj.run(res=res, x=x, min=min, max=max)


@pytest.mark.api_base_clip_parameters
def test_clip5():
    """
    min =np(Tensor)
    """
    x = randtool("float", -1, 10, [3, 3])
    min = np.array([1])
    max = None
    res = np.clip(a=x, a_min=min, a_max=max)
    obj.run(res=res, x=x, min=min, max=max)


@pytest.mark.api_base_clip_parameters
def test_clip6():
    """
    max < x,min=None
    """
    x = randtool("float", 1, 10, [3, 3, 3])
    min = None
    max = 5
    res = np.clip(a=x, a_min=min, a_max=max)
    obj.base(res=res, x=x, min=min, max=max)


@pytest.mark.api_base_clip_parameters
def test_clip7():
    """
    max = x,min=None
    """
    x = np.ones([1, 2, 3])
    min = None
    max = 1
    res = np.clip(a=x, a_min=min, a_max=max)
    obj.base(res=res, x=x, min=min, max=max)


@pytest.mark.api_base_clip_parameters
def test_clip8():
    """
    min=None,max<0
    """
    x = np.array([-10, 2, 0])
    min = None
    max = -1
    res = np.clip(a=x, a_min=min, a_max=max)
    obj.run(res=res, x=x, min=min, max=max)


@pytest.mark.api_base_clip_parameters
def test_clip9():
    """
    max =np(Tensor)
    """
    x = randtool("float", -1, 10, [3, 3])
    min = 1
    max = np.array([2])
    res = np.clip(a=x, a_min=min, a_max=max)
    obj.run(res=res, x=x, min=min, max=max)


@pytest.mark.api_base_clip_parameters
def test_clip10():
    """
    max,min =np(Tensor),min<x<max
    """
    x = randtool("float", -1, 10, [3, 3])
    min = np.array([-10])
    max = np.array([20])
    res = np.clip(a=x, a_min=min, a_max=max)
    obj.run(res=res, x=x, min=min, max=max)


@pytest.mark.api_base_clip_parameters
def test_clip11():
    """
    max,min =np(Tensor),min=x=max
    """
    x = np.ones([1, 2])
    min = np.ones([1])
    max = np.ones([1])
    res = np.clip(a=x, a_min=min, a_max=max)
    obj.run(res=res, x=x, min=min, max=max)


# def test_clip12():
#     """
#     max,min =np(Tensor[]),[]?None,icafe
#     """
#     x = np.array([])
#     min = np.zeros([1])
#     max = np.ones([1])
#     res = np.clip(a=x, a_min=min, a_max=max)
#     obj.run(res=res, x=x, min=min, max=max)


@pytest.mark.api_base_clip_parameters
def test_clip13():
    """
    max_type=np.int32
    """
    x = np.array([[1, 2], [3, 4]])
    t = np.int32
    min = np.array([1.1]).astype(t)
    max = np.array([2.2]).astype(t)
    res = np.array([[1, 2], [2, 2]])
    obj.run(res=res, x=x, min=min, max=max)


@pytest.mark.api_base_clip_parameters
def test_clip14():
    """
    max_type=np.float32,icafe
    """
    x = np.array([[1, 2], [3, 4]])
    t = np.float32
    min = np.array([1.1]).astype(t)
    max = np.array([2.2]).astype(t)
    res = np.array([[1.1, 2], [2.2, 2.2]])
    obj.run(res=res, x=x, min=min, max=max)


@pytest.mark.api_base_clip_parameters
def test_clip15():
    """
    max_type=np.float64
    """
    x = np.array([[1, 2], [3, 4]])
    t = np.float64
    min = np.array([1.1]).astype(t)
    max = np.array([2.2]).astype(t)
    res = np.array([[1.1, 2], [2.2, 2.2]])
    obj.run(res=res, x=x, min=min, max=max)


@pytest.mark.api_base_clip_parameters
def test_clip16():
    """
    name = "test name"
    """
    x = np.array([1, 2])
    min = np.zeros([1])
    max = np.ones([1])
    name = "test name"
    res = np.clip(a=x, a_min=min, a_max=max)
    obj.run(res=res, x=x, min=min, max=max, name=name)


@pytest.mark.api_base_clip_parameters
def test_clip17():
    """
    max,min =np(Tensor_shape>1)
    """
    x = np.array([1, 2, 3])
    min = np.array([1])
    max = np.array([2, 2])
    res = np.clip(a=x, a_min=min, a_max=np.array([2]))
    obj.run(res=res, x=x, min=min, max=max)


@pytest.mark.api_base_clip_parameters
def test_clip18():
    """
    no min max
    """
    x = np.array([1, 2, 3])
    obj.run(res=x, x=x)


@pytest.mark.api_base_clip_parameters
def test_clip19():
    """
    max_type=np.int32
    tensor.clip
    """
    paddle.disable_static()
    x = np.array([[1, 2], [3, 4]]).astype(np.int32)
    x_tensor = paddle.to_tensor(x)
    t = np.int32
    min = paddle.to_tensor(np.array([1.1]).astype(t))
    max = paddle.to_tensor(np.array([2.2]).astype(t))
    out = x_tensor.clip(min, max)
    res = np.array([[1, 2], [2, 2]])
    compare(res, out)


@pytest.mark.api_base_clip_parameters
def test_clip20():
    """
    max_type=np.int64
    tensor.clip
    """
    paddle.disable_static()
    x = np.array([[1, 2], [3, 4]]).astype(np.int64)
    x_tensor = paddle.to_tensor(x)
    # t = np.int32
    min = 1
    max = 2
    out = x_tensor.clip(min, max)
    res = np.array([[1, 2], [2, 2]])
    compare(res, out)
