#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test randn
"""
from apibase import APIBase
import paddle
import pytest
from paddle import fluid
import numpy as np


class TestRandn(APIBase):
    """
    test
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32, np.float64]
        self.places = [fluid.CPUPlace()]
        # self.debug = True
        # self.static = True
        # enable check grad
        self.enable_backward = False
        self.no_grad_var = ["shape"]


obj = TestRandn(paddle.randn)


@pytest.mark.api_base_randn_vartype
def test_randn_base():
    """
    base
    """
    res = np.array([[0.92102236, 0.17973621], [-0.9943417, -1.7103449]])
    obj.base(res=res, shape=[2, 2])


@pytest.mark.api_base_randn_parameters
def test_randn():
    """
    dtype is float32
    """
    res = np.array([[0.92102236, 0.17973621], [-0.9943417, -1.7103449]])
    obj.run(res=res, shape=[2, 2], dtype=np.float32)


@pytest.mark.api_base_randn_exception
def test_randn1():
    """
    exception dtype is int32
    """
    obj.seed = 1
    # res = np.array([[0, 7], [6, 0]])
    obj.exception(etype="NotFoundError", shape=[2, 2], dtype=np.int32)


@pytest.mark.api_base_randn_parameters
def test_randn2():
    """
    seed = 1
    """
    obj.seed = 1
    res = np.array([[-0.30557564, 0.11855337], [0.41220093, -0.09968963]])
    obj.run(res=res, shape=[2, 2], dtype=np.float32)


@pytest.mark.api_base_randn_parameters
def test_randn3():
    """
    shape is tuple
    """
    obj.seed = 1
    res = np.array([[-0.30557564, 0.11855337], [0.41220093, -0.09968963]])
    obj.run(res=res, shape=(2, 2))


@pytest.mark.api_base_randn_parameters
def test_randn4():
    """
    shape is tensor
    """
    obj.seed = 1
    res = np.array([[-0.30557564, 0.11855337], [0.41220093, -0.09968963]])
    obj.run(res=res, shape=np.array([2, 2]))


@pytest.mark.api_base_randn_parameters
def test_randn5():
    """
    gpu
    """
    obj.places = [fluid.CUDAPlace(0)]
    obj.seed = 1
    res = np.array([[-4.080414, -1.3719953], [0.2568497, 1.2351426]])
    obj.run(res=res, shape=np.array([2, 2]))
