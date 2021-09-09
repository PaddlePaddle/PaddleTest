#!/usr/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python

"""
test_mv.py
"""

from apibase import APIBase

import paddle
import pytest
import numpy as np


class TestMv(APIBase):
    """
    test mv
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32, np.float64]
        # self.debug = True
        # enable check grad
        # self.enable_backward = False


obj = TestMv(paddle.mv)
seed = 33


@pytest.mark.api_base_mv_vartype
def test_mv_base():
    """
    base
    """
    x = np.random.uniform(-10, 10, [3, 2])
    vec = np.random.uniform(-10, 10, [2])
    res = np.array([39.25061795, -26.34643373, -128.26865193])
    obj.base(res=res, x=x, vec=vec)


@pytest.mark.api_base_mv_parameters
def test_mv1():
    """
    x_shape=[2,3],vec_shape=[3]
    """
    x = np.array([[2, 1, 3], [3, 0, 1]])
    vec = np.array([3, 5, 1])
    res = np.array([14, 10])
    obj.run(res=res, x=x, vec=vec)


@pytest.mark.api_base_mv_parameters
def test_mv2():
    """
    x_shape=[1,2],vec_shape=[2]
    """
    x = np.array([[-2, 1]])
    vec = np.array([3, 0])
    res = np.array([-6])
    obj.run(res=res, x=x, vec=vec)


@pytest.mark.api_base_mv_parameters
def test_mv3():
    """
    x_shape=[2,1],vec_shape=[2]
    """
    x = np.array([[2], [-1]])
    print("-------x-----", x.shape)

    vec = np.array([3])
    res = np.array([6, -3])
    obj.run(res=res, x=x, vec=vec)


@pytest.mark.api_base_mv_parameters
def test_mv4():
    """
    x_value=1,vec_value=0
    """
    x = np.ones([3, 3])
    vec = np.zeros([3])
    res = np.zeros([3])
    obj.run(res=res, x=x, vec=vec)


@pytest.mark.api_base_mv_exception
def test_mv5():
    """
    x_shape=3-D
    """
    x = np.ones([3, 2, 3])
    vec = np.zeros([3])
    obj.exception(mode="c", etype="InvalidArgument", x=x, vec=vec)


@pytest.mark.api_base_mv_exception
def test_mv6():
    """
    x_shape=[],vec_shape=[]
    """
    x = np.ones([])
    vec = np.zeros([])
    obj.exception(mode="c", etype="InvalidArgument", x=x, vec=vec)


@pytest.mark.api_base_mv_exception
def test_mv7():
    """
    vec_shape=2-D
    """
    x = np.ones([3, 3])
    vec = np.zeros([3, 4])
    obj.exception(mode="c", etype="InvalidArgument", x=x, vec=vec)
