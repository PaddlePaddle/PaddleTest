#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test rand
"""
from apibase import APIBase
import paddle
import pytest
from paddle import fluid
import numpy as np


class TestRand(APIBase):
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


obj = TestRand(paddle.rand)


@pytest.mark.api_base_rand_vartype
def test_rand_base():
    """
    base
    """
    res = np.array([[0.94635665, 0.99746263], [0.57684416, 0.89377254]])
    obj.base(res=res, shape=[2, 2])


@pytest.mark.api_base_rand_parameters
def test_rand():
    """
    dtype is float32
    """
    res = np.array([[0.94635665, 0.99746263], [0.57684416, 0.89377254]])
    obj.run(res=res, shape=[2, 2], dtype=np.float32)


@pytest.mark.api_base_rand_exception
def test_rand1():
    """
    exception dtype is int32 BUG
    """
    obj.seed = 1
    # res = np.array([[0, 7], [6, 0]])
    obj.exception(etype="NotFoundError", shape=[2, 2], dtype=np.int32)


@pytest.mark.api_base_rand_parameters
def test_rand2():
    """
    seed = 1
    """
    obj.seed = 1
    res = np.array([[0.01787627, 0.7649231], [0.67605734, 0.04620579]])
    obj.run(res=res, shape=[2, 2], dtype=np.float32)


@pytest.mark.api_base_rand_parameters
def test_rand3():
    """
    shape is tuple
    """
    obj.seed = 33
    res = np.array([[0.94635665, 0.99746263], [0.57684416, 0.89377254]])
    obj.run(res=res, shape=(2, 2))


@pytest.mark.api_base_rand_parameters
def test_rand4():
    """
    shape is tensor
    """
    obj.seed = 33
    res = np.array([[0.94635665, 0.99746263], [0.57684416, 0.89377254]])
    obj.run(res=res, shape=np.array([2, 2]))


@pytest.mark.api_base_rand_parameters
def test_rand5():
    """
    test gpu
    """
    obj.places = [fluid.CUDAPlace(0)]
    obj.seed = 33
    res = np.array([[7.4177142e-04, 8.0607080e-01], [8.4463596e-01, 4.2317215e-01]])
    obj.run(res=res, shape=np.array([2, 2]))
