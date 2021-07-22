#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test log
"""
from apibase import APIBase
import paddle
import pytest
import numpy as np


class TestLog(APIBase):
    """
    test log
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32, np.float64]
        # log has backward compute
        self.enable_backward = True


obj = TestLog(paddle.log)


@pytest.mark.api_base_log_vartype
def test_log_base():
    """
    base
    """
    x = np.array([[1, 2], [3, 4]])
    res = np.log(x)
    obj.base(res=res, x=x)


@pytest.mark.api_base_log_exception
def test_log1():
    """
    x = float(not tensor)
    """
    x = 3.4
    obj.exception(mode="c", etype="InvalidArgumentError", x=x)


# def test_log2():
#     """
#     x = [], returns none in static mode, is a common problem
#     """
#     x = np.array([])
#     res = np.log(x)
#     obj.run(res=res, x=x)


@pytest.mark.api_base_log_parameters
def test_log3():
    """
    x = large num
    """
    x = np.array([23333, 463333, 665432222])
    res = np.log(x)
    obj.run(res=res, x=x)


@pytest.mark.api_base_log_parameters
def test_log4():
    """
    x = float(tensor)
    """
    x = np.array([0.33332, 0.800002, 0.44444])
    res = np.log(x)
    obj.run(res=res, x=x)


@pytest.mark.api_base_log_parameters
def test_log5():
    """
    x = many dimensions
    """
    x = 1 + np.arange(12).reshape(2, 2, 3)
    res = np.log(x)
    obj.run(res, x=x)


@pytest.mark.api_base_log_parameters
def test_log6():
    """
    name is defined
    """
    x = 1 + np.arange(12).reshape(2, 2, 3)
    res = np.log(x)
    obj.run(res, x=x, name="test_log")


@pytest.mark.api_base_log_parameters
def test_log7():
    """
    x=0
    """
    x = np.array([0])
    res = np.array([-np.inf])
    obj1.run(res=res, x=x)


class TestLog1(APIBase):
    """
    test log
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32, np.float64]
        # if input is invalid, cannot compute backward
        self.enable_backward = False


obj1 = TestLog1(paddle.log)


@pytest.mark.api_base_log_parameters
def test_log8():
    """
    x = negtive num
    """
    x = np.array([-3])
    res = np.array([np.nan])
    obj1.run(res=res, x=x)


class TestLog2(APIBase):
    """
    test log
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.int32]
        # log has backward compute
        self.enable_backward = True


obj2 = TestLog2(paddle.log)


@pytest.mark.api_base_log_exception
def test_log9():
    """
    x = float16(tensor)
    """
    x = np.array([[1, 2], [3, 4]])
    obj2.exception(mode="c", etype="NotFoundError", x=x)
