#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test full_like
"""
from apibase import APIBase
import paddle
import pytest
import numpy as np


class TestFullLike(APIBase):
    """
    test full_like
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.bool, np.float16, np.float32, np.float64, np.int32, np.int64]
        # self.debug = True
        # self.static = True
        # enable check grad
        self.enable_backward = False


obj = TestFullLike(paddle.full_like)


@pytest.mark.api_base_full_like_vartype
def test_full_like_base():
    """
    base
    """
    x = np.array([2, 3])
    fill_value = 1
    res = np.full_like(x, fill_value)
    obj.base(res=res, x=x, fill_value=fill_value)


@pytest.mark.api_base_full_like_parameters
def test_full_like1():
    """
    fill_value is bool, bool:True, dtype:bool
    """
    x = np.array([2, 3])
    fill_value = True
    res = np.full_like(x, fill_value)
    obj.run(res=res, x=x, fill_value=fill_value, dtype=np.bool)


@pytest.mark.api_base_full_like_parameters
def test_full_like2():
    """
    fill_value is bool, bool:False, dtype:bool
    """
    x = np.array([2, 3])
    fill_value = False
    res = np.full_like(x, fill_value)
    obj.run(res=res, x=x, fill_value=fill_value, dtype=np.bool)


@pytest.mark.api_base_full_like_parameters
def test_full_like3():
    """
    fill_value is float, dtype:float16
    """
    x = np.array([2, 3])
    fill_value = 1.0
    res = np.full_like(x, fill_value).astype(np.float16)
    obj.run(res=res, x=x, fill_value=fill_value, dtype=np.float16)


@pytest.mark.api_base_full_like_parameters
def test_full_like4():
    """
    fill_value is float, dtype:float32
    """
    x = np.array([2, 3])
    fill_value = 1.0
    res = np.full_like(x, fill_value)
    obj.run(res=res, x=x, fill_value=fill_value, dtype=np.float32)


@pytest.mark.api_base_full_like_parameters
def test_full_like5():
    """
    fill_value is int, dtype:int32
    """
    x = np.array([2, 3])
    fill_value = 1
    res = np.full_like(x, fill_value)
    obj.run(res=res, x=x, fill_value=fill_value, dtype=np.int32)


@pytest.mark.api_base_full_like_parameters
def test_full_like6():
    """
    fill_value is int, dtype:int64
    """
    x = np.array([2, 3])
    fill_value = 1
    res = np.full_like(x, fill_value)
    obj.run(res=res, x=x, fill_value=fill_value, dtype=np.int64)


@pytest.mark.api_base_full_like_parameters
def test_full_like7():
    """
    fill_value is bool, bool:True, dtype:str(bool)
    """
    x = np.array([2, 3])
    fill_value = True
    res = np.full_like(x, fill_value)
    obj.run(res=res, x=x, fill_value=fill_value, dtype="bool")


@pytest.mark.api_base_full_like_parameters
def test_full_like8():
    """
    fill_value is bool, bool:False, dtype:str(bool)
    """
    x = np.array([2, 3])
    fill_value = False
    res = np.full_like(x, fill_value)
    obj.run(res=res, x=x, fill_value=fill_value, dtype="bool")


@pytest.mark.api_base_full_like_parameters
def test_full_like9():
    """
    fill_value is float, dtype:str(float16)
    """
    x = np.array([2, 3])
    fill_value = 1.0
    res = np.full_like(x, fill_value)
    obj.run(res=res, x=x, fill_value=fill_value, dtype="float16")


@pytest.mark.api_base_full_like_parameters
def test_full_like10():
    """
    fill_value is float, dtype:str(float32)
    """
    x = np.array([2, 3])
    fill_value = 1.0
    res = np.full_like(x, fill_value)
    obj.run(res=res, x=x, fill_value=fill_value, dtype="float32")


@pytest.mark.api_base_full_like_parameters
def test_full_like11():
    """
    fill_value is int, dtype:str(int32)
    """
    x = np.array([2, 3])
    fill_value = 1
    res = np.full_like(x, fill_value)
    obj.run(res=res, x=x, fill_value=fill_value, dtype="int32")


@pytest.mark.api_base_full_like_parameters
def test_full_like12():
    """
    fill_value is int, dtype:str(int64)
    """
    x = np.array([2, 3])
    fill_value = 1
    res = np.full_like(x, fill_value)
    obj.run(res=res, x=x, fill_value=fill_value, dtype="int64")


@pytest.mark.api_base_full_like_parameters
def test_full_like13():
    """
    fill_value is inf
    """
    x = np.array([2, 3])
    fill_value = float("inf")
    res = np.full_like(x, fill_value)
    obj.run(res=res, x=x, fill_value=fill_value, dtype="int64")


@pytest.mark.api_base_full_like_parameters
def test_full_like14():
    """
    fill_value is -inf
    """
    x = np.array([2, 3])
    fill_value = float("-inf")
    res = np.full_like(x, fill_value)
    obj.run(res=res, x=x, fill_value=fill_value, dtype="int64")
