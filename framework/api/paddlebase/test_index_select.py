#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test index_select
"""
from apibase import APIBase
import paddle
import pytest
import numpy as np


class TestIndexSelect(APIBase):
    """
    test index_select
    """

    def hook(self):
        """
        implement
        """
        # x only support float32,float64,int32,int64 tensor
        self.types = [np.float32, np.float64, np.int32, np.int64]
        self.enable_backward = False
        self.no_grad_var = ["index"]


obj = TestIndexSelect(paddle.index_select)


@pytest.mark.api_base_index_select_vartype
def test_index_select_base():
    """
    base
    axis use default value
    """
    x = np.arange(6).reshape(2, 3)
    index = np.array([0]).astype("int32")
    res = np.arange(3).reshape(1, 3)
    obj.base(res=res, x=x, index=index)


@pytest.mark.api_base_index_select_parameters
def test_index_select1():
    """
    axis=1
    """
    x = np.arange(6).reshape(2, 3)
    index = np.array([0, 2]).astype("int32")
    axis = 1
    res = np.array([[0, 2], [3, 5]])
    obj.run(res=res, x=x, index=index, axis=axis)


@pytest.mark.api_base_index_select_parameters
def test_index_select2():
    """
    index=int64(tensor)
    """
    x = np.arange(6).reshape(2, 3)
    index = np.array([0, 2]).astype("int64")
    axis = 1
    res = np.array([[0, 2], [3, 5]])
    obj.run(res=res, x=x, index=index, axis=axis)


@pytest.mark.api_base_index_select_exception
def test_index_select3():
    """
    index=float32(tensor)
    """
    x = np.arange(6).reshape(2, 3)
    index = np.array([0, 2]).astype("float32")
    axis = 1
    obj.exception(mode="c", etype="InvalidArgument", x=x, index=index, axis=axis)


@pytest.mark.api_base_index_select_exception
def test_index_select4():
    """
    index=int32(not tensor)
    """
    x = np.arange(6).reshape(2, 3)
    index = 0
    axis = 1
    obj.exception(mode="c", etype="InvalidArgument", x=x, index=index, axis=axis)


@pytest.mark.api_base_index_select_exception
def test_index_select5():
    """
    index doesn't exist
    """
    x = np.arange(6).reshape(2, 3)
    index = np.array([0, 5]).astype("int64")
    axis = 1
    obj.exception(mode="c", etype="InvalidArgument", x=x, index=index, axis=axis)


@pytest.mark.api_base_index_select_exception
def test_index_select6():
    """
    axis doesn't exist
    """
    x = np.arange(6).reshape(2, 3)
    index = np.array([0, 2]).astype("int64")
    axis = 2
    obj.exception(mode="c", etype="OutOfRange", x=x, index=index, axis=axis)


class TestIndexSelect1(APIBase):
    """
    test index_select
    """

    def hook(self):
        """
        implement
        """
        # index only support int32 and int64 tensor
        self.types = [np.int32, np.int64]
        self.enable_backward = False
        self.no_grad_var = ["x"]


obj1 = TestIndexSelect1(paddle.index_select)


@pytest.mark.api_base_index_select_exception
def test_index_select7():
    """
    x=int32(not tensor)
    """
    x = 32
    index = np.array([0])
    axis = 1
    obj1.exception(mode="c", etype="InvalidArgument", x=x, index=index, axis=axis)


@pytest.mark.api_base_index_select_exception
def test_index_select8():
    """
    x=float16(tensor)
    """
    x = np.arange(6).reshape(2, 3).astype("float16")
    index = np.array([0, 2])
    axis = 1
    obj1.exception(mode="c", etype="NotFound", x=x, index=index, axis=axis)


@pytest.mark.api_base_index_select_exception
def test_index_select9():
    """
    axis=float
    """
    x = np.arange(6).reshape(2, 3).astype("int32")
    index = np.array([0, 2])
    axis = 1.3
    obj1.exception(mode="c", etype="InvalidArgument", x=x, index=index, axis=axis)
