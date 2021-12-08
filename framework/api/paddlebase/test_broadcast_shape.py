#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test broadcast_shape
"""
from apibase import APIBase
import paddle
import pytest
import numpy as np


@pytest.mark.api_base_broadcast_shape_vartype
def test_broadcast_shape_base():
    """
    x_shape = [2, 1, 3]
    y_shape = [1, 3, 1]
    """
    x_shape = [2, 1, 3]
    y_shape = [1, 3, 1]
    shape = paddle.broadcast_shape(x_shape, y_shape)
    assert isinstance(shape, list)
    assert shape == [2, 3, 3]


@pytest.mark.api_base_broadcast_shape_parameters
def test_broadcast_shape1():
    """
    x_shape = [2, 1, 3, 1, 4, 2, 3]
    y_shape = [1, 3, 1, 2, 1, 1, 1]
    """
    x_shape = [2, 1, 3, 1, 4, 2, 3]
    y_shape = [1, 3, 1, 2, 1, 1, 1]
    shape = paddle.broadcast_shape(x_shape, y_shape)
    assert isinstance(shape, list)
    assert shape == [2, 3, 3, 2, 4, 2, 3]


@pytest.mark.api_base_broadcast_shape_parameters
def test_broadcast_shape2():
    """
    x_shape = [2, 1, 3, 1, 4, 2, 3]
    y_shape = [1, 3, 1, 2, 1, 1, 1]
    """
    x_shape = [2, 1, 3, 1, 4, 2, 3]
    y_shape = [1, 3, 1, 2, 1, 1, 1]
    shape = paddle.broadcast_shape(x_shape, y_shape)
    assert isinstance(shape, list)
    assert shape == [2, 3, 3, 2, 4, 2, 3]


@pytest.mark.api_base_broadcast_shape_parameters
def test_broadcast_shape3():
    """
    x_shape = [2, 1, 3, 1, 4, 2, 3]
    y_shape = [1, 3, 1, 2, 1, 2, 3]
    """
    x_shape = [2, 1, 3, 1, 4, 2, 3]
    y_shape = [1, 3, 1, 2, 1, 2, 3]
    shape = paddle.broadcast_shape(x_shape, y_shape)
    assert isinstance(shape, list)
    assert shape == [2, 3, 3, 2, 4, 2, 3]


@pytest.mark.api_base_broadcast_shape_parameters
def test_broadcast_shape4():
    """
    x_shape = [2, 1, 3, 1, 4, 2, 3]
    y_shape = [3, 1, 2, 1, 1, 3]
    """
    x_shape = [2, 1, 3, 1, 4, 2, 3]
    y_shape = [3, 1, 2, 1, 1, 3]
    shape = paddle.broadcast_shape(x_shape, y_shape)
    assert isinstance(shape, list)
    assert shape == [2, 3, 3, 2, 4, 2, 3]


@pytest.mark.api_base_broadcast_shape_parameters
def test_broadcast_shape5():
    """
    x_shape = [2, 1, 3, 1, 4, 2, 3]
    y_shape = [1, 2, 1, 1, 3]
    """
    x_shape = [2, 1, 3, 1, 4, 2, 3]
    y_shape = [1, 2, 1, 1, 3]
    shape = paddle.broadcast_shape(x_shape, y_shape)
    assert isinstance(shape, list)
    assert shape == [2, 1, 3, 2, 4, 2, 3]


@pytest.mark.api_base_broadcast_shape_parameters
def test_broadcast_shape6():
    """
    x_shape = [2, 1, 3, 1, 4, 2, 3, 7, 2]
    y_shape = [1, 2, 1, 1, 2]
    """
    x_shape = [2, 1, 3, 1, 4, 2, 3, 7, 2]
    y_shape = [1, 2, 1, 1, 2]
    shape = paddle.broadcast_shape(x_shape, y_shape)
    assert isinstance(shape, list)
    assert shape == [2, 1, 3, 1, 4, 2, 3, 7, 2]
