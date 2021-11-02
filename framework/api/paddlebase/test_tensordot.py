#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_tensordot
"""
from apibase import APIBase
from apibase import randtool
import paddle
import pytest
import numpy as np


class TestTensorDot(APIBase):
    """
    test
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32, np.float64]
        self.delta = 1e-2
        # self.debug = True
        # self.static = True
        # enable check grad
        # self.enable_backward = True


obj = TestTensorDot(paddle.tensordot)


@pytest.mark.api_base_tensordot_vartype
def test_tensordot_base():
    """
    base
    """
    x = randtool("float", -10, 10, [3, 3])
    y = randtool("float", 4, 14, [3, 3])
    res = np.tensordot(x, y, axes=2).reshape(1)
    obj.base(res=res, x=x, y=y, axes=2)


@pytest.mark.api_base_tensordot_parameters
def test_tensordot0():
    """
    default
    """
    x = randtool("int", -10, 10, [2, 3, 3, 4])
    y = randtool("float", -4, -1, [3, 4, 3, 4])
    res = np.tensordot(x, y)
    obj.run(res=res, x=x, y=y)


@pytest.mark.api_base_tensordot_parameters
def test_tensordot1():
    """
    axes = 0
    """
    x = randtool("int", -10, 10, [2, 3, 3, 4])
    y = randtool("float", -4, -1, [3, 4, 3, 4])
    res = np.tensordot(x, y, axes=0)
    obj.run(res=res, x=x, y=y, axes=0)


@pytest.mark.api_base_tensordot_parameters
def test_tensordot2():
    """
    axes = 1
    """
    x = randtool("int", -10, 10, [3, 4, 3, 4])
    y = randtool("float", -4, -1, [4, 4, 3, 4])
    res = np.tensordot(x, y, axes=1)
    obj.run(res=res, x=x, y=y, axes=1)


@pytest.mark.api_base_tensordot_parameters
def test_tensordot3():
    """
    axes = 3
    """
    x = randtool("int", -10, 10, [2, 3, 4, 4])
    y = randtool("float", -4, -1, [3, 4, 4, 4])
    res = np.tensordot(x, y, axes=0)
    obj.run(res=res, x=x, y=y, axes=0)


@pytest.mark.api_base_tensordot_parameters
def test_tensordot4():
    """
    axes = 4
    """
    x = randtool("int", -10, 10, [2, 3, 4, 4])
    y = randtool("float", -4, -1, [2, 3, 4, 4])
    res = np.tensordot(x, y, axes=0)
    obj.run(res=res, x=x, y=y, axes=0)


@pytest.mark.api_base_tensordot_parameters
def test_tensordot5():
    """
    axes: list
    """
    x = randtool("int", -10, 10, [2, 3, 4, 2])
    y = randtool("float", -4, -1, [2, 3, 8, 7])
    axes = [0, 1]
    res = np.tensordot(x, y, axes=([0, 1], [0, 1]))
    obj.run(res=res, x=x, y=y, axes=axes)


@pytest.mark.api_base_tensordot_parameters
def test_tensordot6():
    """
    axes: tuple
    """
    x = randtool("int", -10, 10, [2, 3, 4, 2])
    y = randtool("float", -4, -1, [8, 7, 4, 2])
    axes = [2, 3]
    res = np.tensordot(x, y, axes=([2, 3], [2, 3]))
    obj.run(res=res, x=x, y=y, axes=axes)


@pytest.mark.api_base_tensordot_parameters
def test_tensordot7():
    """
    axes: tuple
    """
    x = randtool("int", -10, 10, [2, 3, 4, 2])
    y = randtool("float", -4, -1, [8, 7, 4, 2])
    axes = [2, 3]
    res = np.tensordot(x, y, axes=([2, 3], [2, 3]))
    obj.run(res=res, x=x, y=y, axes=axes)


@pytest.mark.api_base_tensordot_parameters
def test_tensordot8():
    """
    axes: tuple(list)
    """
    x = randtool("int", -10, 10, [2, 3, 4, 2])
    y = randtool("float", -4, -1, [8, 7, 4, 2])
    axes = [2, 3]
    res = np.tensordot(x, y, axes=([2, 3], [2, 3]))
    obj.run(res=res, x=x, y=y, axes=axes)


@pytest.mark.api_base_tensordot_parameters
def test_tensordot9():
    """
    axes: list[tuple]
    """
    x = randtool("int", -10, 10, [2, 7, 4, 2])
    y = randtool("float", -4, -1, [8, 7, 4, 2])
    axes = [(1, 2, 3)]
    res = np.tensordot(x, y, axes=([1, 2, 3], [1, 2, 3]))
    obj.run(res=res, x=x, y=y, axes=axes)


@pytest.mark.api_base_tensordot_parameters
def test_tensordot10():
    """
    axes: list[tuple, tuple]
    """
    x = randtool("int", -10, 10, [2, 7, 4, 2])
    y = randtool("float", -4, -1, [7, 7, 4, 2])
    axes = [(1, 2, 3), (0, 2, 3)]
    res = np.tensordot(x, y, axes=([1, 2, 3], [0, 2, 3]))
    obj.run(res=res, x=x, y=y, axes=axes)


@pytest.mark.api_base_tensordot_parameters
def test_tensordot11():
    """
    axes: tuple(list, list)
    """
    x = randtool("int", -10, 10, [2, 7, 4, 2])
    y = randtool("float", -4, -1, [7, 7, 4, 2])
    axes = ([1, 2, 3], [0, 2, 3])
    res = np.tensordot(x, y, axes=([1, 2, 3], [0, 2, 3]))
    obj.run(res=res, x=x, y=y, axes=axes)


@pytest.mark.api_base_tensordot_parameters
def test_tensordot12():
    """
    axes: tuple(list, list, ...), len() > 2
    """
    x = randtool("int", -10, 10, [2, 7, 4, 2])
    y = randtool("float", -4, -1, [7, 7, 4, 2])
    axes = ([1, 2, 3], [0, 2, 3], [4, 5, 6])
    res = np.tensordot(x, y, axes=([1, 2, 3], [0, 2, 3]))
    obj.run(res=res, x=x, y=y, axes=axes)


@pytest.mark.api_base_tensordot_parameters
def test_tensordot13():
    """
    y: broadcast
    """
    x = randtool("float", -10, 10, [4, 3, 5])
    y = randtool("int", -4, -1, [4, 1, 8])
    y1 = np.repeat(y, 3, axis=1)
    axes = [0, 1]
    res = np.tensordot(x, y1, axes=([0, 1], [0, 1]))
    obj.run(res=res, x=x, y=y, axes=axes)


@pytest.mark.api_base_tensordot_parameters
def test_tensordot14():
    """
    x: broadcast
    """
    x = randtool("float", -10, 10, [2, 4, 3, 5])
    y = randtool("int", -4, -1, [2, 4, 1, 5])
    x1 = np.broadcast_to(x, (2, 4, 3, 5))
    y1 = np.broadcast_to(y, (2, 4, 3, 5))
    axes = [0, 2]
    res = np.tensordot(x1, y1, axes=([0, 2], [0, 2]))
    obj.run(res=res, x=x, y=y, axes=axes)


@pytest.mark.api_base_tensordot_parameters
def test_tensordot15():
    """
    axes: broadcast
    """
    x = randtool("float", -10, 10, [4, 2, 3, 5, 4])
    y = randtool("int", -4, -1, [2, 4, 3, 5, 8])
    axes = [[0, 1, 2, 3], [1, 0]]
    res = np.tensordot(x, y, axes=([0, 1, 2, 3], [1, 0, 2, 3]))
    obj.run(res=res, x=x, y=y, axes=axes)
