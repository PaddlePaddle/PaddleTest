#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test is_empty
"""
import paddle
import numpy as np
import pytest


@pytest.mark.api_base_is_empty_vartype
def test_is_empty():
    """
    x is tensor(np.array), False
    """
    x = np.arange(1, 7).reshape((2, 3)).astype(np.float32)
    x_tensor = paddle.to_tensor(x)
    check = paddle.is_empty(x_tensor)
    assert check == False


@pytest.mark.api_base_is_empty_vartype
def test_is_empty1():
    """
    x is tensor(list), False
    """
    x = [1, 2, 3]
    x_tensor = paddle.to_tensor(x)
    check = paddle.is_empty(x_tensor)
    assert check == False


@pytest.mark.api_base_is_empty_vartype
def test_is_empty2():
    """
    x is tensor(num), False
    """
    x = 8
    x_tensor = paddle.to_tensor(x)
    check = paddle.is_empty(x_tensor)
    assert check == False


@pytest.mark.api_base_is_empty_parameters
def test_is_empty3():
    """
    x is tensor([]), True
    """
    x = []
    x_tensor = paddle.to_tensor(x)
    check = paddle.is_empty(x_tensor)
    assert check == True


@pytest.mark.api_base_is_empty_parameters
def test_is_empty4():
    """
    x is tensor(np.array([[]] * 3)), True
    """
    x = np.array([[]] * 3)
    x_tensor = paddle.to_tensor(x)
    check = paddle.is_empty(x_tensor)
    assert check == True


@pytest.mark.api_base_is_empty_parameters
def test_is_empty5():
    """
    x is tensor(np.array([[]] * 3)), True
    """
    x = np.array([[]] * 3).astype(np.int32)
    x_tensor = paddle.to_tensor(x)
    check = paddle.is_empty(x_tensor)
    assert check == True


@pytest.mark.api_base_is_empty_parameters
def test_is_empty6():
    """
    x is tensor(np.array([[]] * 3)), True
    """
    x = np.array([[]] * 3).astype(np.int64)
    x_tensor = paddle.to_tensor(x)
    check = paddle.is_empty(x_tensor)
    assert check == True


@pytest.mark.api_base_is_empty_vartype
def test_is_empty5():
    """
    x is tensor(np.array([[]] * 3)), True
    """
    x = np.array([[]] * 3).astype(np.float64)
    x_tensor = paddle.to_tensor(x)
    check = paddle.is_empty(x_tensor)
    assert check == True
