#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_dist
"""
from apibase import APIBase
from apibase import randtool
import paddle
import pytest
import numpy as np


class TestDist(APIBase):
    """
    test
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32, np.float64]
        # self.debug = True
        # self.static = True
        # enable check grad
        # self.enable_backward = False
        # self.delta = 1e-2


obj = TestDist(paddle.dist)


@pytest.mark.api_base_dist_vartype
def test_dist_base():
    """
    base
    default: p = 2
    """
    x = randtool("float", 1, 2, [4, 4])
    y = randtool("float", -1, 1, [4, 4])
    z = x - y
    res = np.linalg.norm(z).reshape([1])
    obj.base(res=res, x=x, y=y)


@pytest.mark.api_base_dist_parameters
def test_dist0():
    """
    x.shape == y.shape
    p = 0
    """
    obj.enable_backward = False
    # x = randtool("float", 1, 2, [4, 4])
    # y = randtool('float', -1, 1, [4, 4])
    x = np.array([[3, 1], [3, 3]])
    y = np.array([[3, 3], [3, 1]])
    # z = x - y
    # res = np.linalg.norm(z, ord=0).reshape([1])
    res = np.array([2])
    obj.run(res=res, x=x, y=y, p=0)


@pytest.mark.api_base_dist_parameters
def test_dist1():
    """
    x.shape == y.shape
    p = inf
    """
    obj.enable_backward = False
    # x = randtool("float", 1, 2, [4, 4])
    # y = randtool('float', -1, 1, [4, 4])
    x = np.array([[3, 100], [3, 3]])
    y = np.array([[3, 0], [3, 1]])
    # z = x - y
    # res = np.linalg.norm(z, ord=0).reshape([1])
    res = np.array([100])
    obj.run(res=res, x=x, y=y, p=float("inf"))


@pytest.mark.api_base_dist_parameters
def test_dist2():
    """
    x.shape == y.shape
    p = -inf
    """
    obj.enable_backward = False
    x = np.array([[3, 100], [3, 3]])
    y = np.array([[3, 0], [3, 5]])
    # z = x - y
    # res = np.linalg.norm(z, ord=0).reshape([1])
    res = np.array([0])
    obj.run(res=res, x=x, y=y, p=float("-inf"))


@pytest.mark.api_base_dist_parameters
def test_dist3():
    """
    x.shape == y.shape
    p = 1
    differernt with np
    """
    obj.enable_backward = True
    x = randtool("float", 1, 2, [4, 4])
    y = randtool("float", -1, 1, [4, 4])
    z = x - y
    res = np.sum(np.abs(z)).reshape([1])
    obj.run(res=res, x=x, y=y, p=1)


@pytest.mark.api_base_dist_parameters
def test_dist4():
    """
    broadcast x.shape.dim: != y.shape.dim
    p = 2
    """
    # obj.enable_backward = False
    x = randtool("float", 1, 2, [2, 1, 4, 4])
    y = randtool("float", -1, 1, [7, 1, 4])
    z = x - y
    res = np.linalg.norm(z).reshape([1])
    obj.run(res=res, x=x, y=y)


@pytest.mark.api_base_dist_parameters
def test_dist5():
    """
    broadcast: x.shape.dim == y.shpe.dim
    p = 2
    """
    # obj.enable_backward = False
    x = randtool("float", 1, 2, [2, 1, 1, 4, 4])
    y = randtool("float", -1, 1, [2, 8, 7, 1, 4])
    z = x - y
    res = np.linalg.norm(z).reshape([1])
    obj.run(res=res, x=x, y=y)


@pytest.mark.api_base_dist_parameters
def test_dist6():
    """
    x, y: vector
    p = 2
    """
    # obj.enable_backward = False
    x = np.random.rand(10)
    y = np.random.randint(-1, 1, (10,))
    z = x - y
    res = np.linalg.norm(z).reshape([1])
    obj.run(res=res, x=x, y=y)


@pytest.mark.api_base_dist_parameters
def test_dist7():
    """
    broadcast: x-->>vector
    p = 2
    """
    # obj.enable_backward = False
    x = np.random.rand(10)
    y = np.random.rand(4, 10)
    z = x - y
    res = np.linalg.norm(z).reshape([1])
    obj.run(res=res, x=x, y=y)


@pytest.mark.api_base_dist_parameters
def test_dist8():
    """
    broadcast
    p = 4
    """
    # obj.enable_backward = False
    x = np.random.rand(10)
    y = np.random.rand(4, 10)
    z = x - y
    z = np.abs(z)
    res = 0
    for i in z.flatten():
        res += i ** 4
    res = res ** 0.25
    res = np.array([res])
    obj.run(res=res, x=x, y=y, p=4)


@pytest.mark.api_base_dist_parameters
def test_dist9():
    """
    broadcast
    p = 7
    """
    # obj.enable_backward = False
    x = np.random.rand(2, 4, 1, 3)
    y = np.random.rand(4, 3, 1)
    z = x - y
    z = np.abs(z)
    res = 0
    for i in z.flatten():
        res += i ** 7
    res = res ** (1 / 7)
    res = np.array([res])
    obj.run(res=res, x=x, y=y, p=7)


@pytest.mark.api_base_dist_exception
def test_dist10():
    """
    could not be broadcast
    """
    # obj.enable_backward = False
    x = np.random.rand(2, 4, 4, 3)
    y = np.random.rand(4, 3, 1)
    obj.exception(etype=ValueError, mode="python", x=x, y=y)
