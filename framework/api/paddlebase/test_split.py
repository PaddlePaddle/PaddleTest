#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_split.py
"""
from apibase import APIBase
import paddle
import pytest
import numpy as np


class TestSplit(APIBase):
    """
    test split
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float16, np.float32, np.float64, np.int32, np.int64, np.int8]
        self.enable_backward = False
        # static_graph of split api is not supported in this frame
        self.static = False
        self.dygraph = True
        self.debug = True


obj = TestSplit(paddle.split)


@pytest.mark.api_base_split_vartype
def test_split_base():
    """
    base
    """
    x = np.arange(6).reshape(2, 3)
    axis = 1
    num_or_sections = 3
    res1 = np.split(x, num_or_sections, axis)
    obj.base(res=res1, x=x, num_or_sections=num_or_sections, axis=axis)


@pytest.mark.api_base_split_parameters
def test_split1():
    """
    num_or_sections=list
    """
    x = np.arange(6).reshape(2, 3)
    axis = -1
    num_or_sections = [2, -1]
    res = np.split(x, [2], axis)
    obj.run(res=res, x=x, num_or_sections=num_or_sections, axis=axis)


@pytest.mark.api_base_split_parameters
def test_split2():
    """
    num_or_sections=tuple
    """
    x = np.arange(6).reshape(2, 3)
    axis = 1
    num_or_sections = (2, 1)
    res = np.split(x, [2], axis)
    obj.run(res=res, x=x, num_or_sections=num_or_sections, axis=axis)


@pytest.mark.api_base_split_exception
def test_split3():
    """
    ValueError:sum of num_or_section>dim
    """
    x = np.arange(6).reshape(2, 3)
    axis = 1
    num_or_sections = [1, 1, 1, 1]
    obj.exception(mode="c", etype="InvalidArgument", x=x, num_or_sections=num_or_sections, axis=axis)


@pytest.mark.api_base_split_parameters
def test_split4():
    """
    axis:use default value
    """
    x = np.arange(6).reshape(2, 3)
    num_or_sections = 2
    res = np.split(x, 2, 0)
    obj.run(res=res, x=x, num_or_sections=num_or_sections)


@pytest.mark.api_base_split_exception
def test_split5():
    """
    TypeError:num_or_sections=float
    """
    x = np.arange(6).reshape(2, 3)
    axis = 1
    num_or_sections = 3.4
    obj.exception(mode="python", etype=TypeError, x=x, num_or_sections=num_or_sections, axis=axis)


class TestSplit1(APIBase):
    """
    test split
    """

    def hook(self):
        """
        implement
        """
        # axis only support int type
        self.types = [np.int32, np.int64]
        self.enable_backward = False
        # Cannot support static_graph of split api in this frame
        self.static = False
        self.dygraph = True
        self.debug = True


obj1 = TestSplit1(paddle.split)


@pytest.mark.api_base_split_parameters
def test_split6():
    """
    axis=tensor
    """
    x = np.arange(6).reshape(2, 3)
    axis = 1 + np.arange(1)
    num_or_sections = 3
    res = np.split(x, num_or_sections, 1)
    obj1.run(res=res, x=x, num_or_sections=num_or_sections, axis=axis)


@pytest.mark.api_base_split_exception
def test_split7():
    """
    TypeError:axis=list
    """
    x = np.arange(6).reshape(2, 3)
    axis = [1]
    num_or_sections = 3
    obj1.exception(mode="python", etype=TypeError, x=x, num_or_sections=num_or_sections, axis=axis)


@pytest.mark.api_base_split_exception
def test_split8():
    """
    TypeError:axis=float
    """
    x = np.arange(6).reshape(2, 3)
    axis = 1.2
    num_or_sections = 3
    obj1.exception(mode="c", etype="InvalidArgument", x=x, num_or_sections=num_or_sections, axis=axis)


@pytest.mark.api_base_split_parameters
def test_split9():
    """
    num_or_sections include 0
    """
    x = np.arange(13)
    paddle.disable_static()
    xp = paddle.to_tensor(x)
    num_or_sections = [5, 4, 0, 4]
    rslt = paddle.split(xp, num_or_sections)
    idx = [0, 5, 9, 9, 13]
    len1 = len(rslt)
    for i in range(len1):
        assert np.allclose(rslt[i], x[idx[i] : idx[i + 1]])


class TestSplit2(APIBase):
    """
    test split
    """

    def hook(self):
        """
        implement
        """
        # test x input type bool
        self.types = [np.bool]
        self.enable_backward = False
        # Cannot support static_graph of split api in this frame
        self.static = False
        self.dygraph = True
        self.debug = True


obj2 = TestSplit2(paddle.split)


@pytest.mark.api_base_split_vartype
def test_split10():
    """
    x=bool
    """
    x = np.array([1, 0, 1])
    axis = 0
    num_or_sections = 3
    res = [np.array([True]), np.array([False]), np.array([True])]
    obj2.run(res=res, x=x, num_or_sections=num_or_sections, axis=axis)


class TestSplit3(APIBase):
    """
    test split
    """

    def hook(self):
        """
        implement
        """
        # test x input type int8
        self.types = [np.int8]
        self.enable_backward = False
        # Cannot support static_graph of split api in this frame
        self.static = False
        self.dygraph = True
        self.debug = True


obj3 = TestSplit3(paddle.split)


# @pytest.mark.api_base_split_exception
# def test_split11():
#    """
#    TypeError:x=int8
#    """
#    x = np.arange(6).reshape(2, 3)
#    axis = 1
#    num_or_sections = 3
#    obj3.exception(mode="c", etype="NotFound", x=x, num_or_sections=num_or_sections, axis=axis)
