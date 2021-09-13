#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_rank.py
"""
from apibase import APIBase
from apibase import randtool
import paddle
import pytest
import numpy as np


class TestRank(APIBase):
    """
    test rank
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.bool, np.int32, np.int64, np.float32, np.float64]
        # self.debug = True
        # self.static = True
        # enable check grad
        self.enable_backward = False


obj = TestRank(paddle.rank)


@pytest.mark.api_base_rank_vartype
def test_rank_base():
    """
    rank base
    """
    x = np.arange(1, 2).reshape((1, 1, 1, 1)).astype(np.int32)
    res = np.array(4)
    obj.base(res=res, input=x)


@pytest.mark.api_base_rank_parameters
def test_rank1():
    """
    rank of x is 4
    """
    x = np.arange(1, 2).reshape((1, 1, 1, 1)).astype(np.int32)
    res = np.array(4)
    obj.run(res=res, input=x)


@pytest.mark.api_base_rank_parameters
def test_rank2():
    """
    rank of x is 4
    """
    x = np.arange(1, 25).reshape((3, 2, 2, 2)).astype(np.float32)
    res = np.array(4)
    obj.run(res=res, input=x)


@pytest.mark.api_base_rank_parameters
def test_rank3():
    """
    rank of x is 2
    """
    x = np.array([[6, 4], [0, 0]]).astype(np.float32)
    res = np.array(2)
    obj.run(res=res, input=x)


@pytest.mark.api_base_rank_parameters
def test_rank4():
    """
    rank of x is 2
    """
    x = np.array([[6, 4], [3, 2]]).astype(np.int32)
    res = np.array(2)
    obj.run(res=res, input=x)


@pytest.mark.api_base_rank_parameters
def test_rank5():
    """
    rank of x is 1
    """
    x = np.arange(1, 5).reshape((4,)).astype(np.int32)
    res = np.array(1)
    obj.run(res=res, input=x)


@pytest.mark.api_base_rank_parameters
def test_rank6():
    """
    rank of x is 2
    """
    x = np.arange(1, 5).reshape((4, 1)).astype(np.int32)
    res = np.array(2)
    obj.run(res=res, input=x)
