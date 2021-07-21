#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test is_tensor
"""
import paddle
import numpy as np
import pytest


@pytest.mark.api_base_is_tensor_vartype
def test_is_tensor():
    """
    x is tensor, True
    """
    x = np.arange(1, 7).reshape((2, 3)).astype(np.float32)
    x_tensor = paddle.to_tensor(x)
    check = paddle.is_tensor(x_tensor)
    assert check == True


@pytest.mark.api_base_is_tensor_parameters
def test_is_tensor1():
    """
    x is np.array, False
    """
    x = np.arange(1, 7).reshape((2, 3)).astype(np.float32)
    check = paddle.is_tensor(x)
    assert check == False


@pytest.mark.api_base_is_tensor_vartype
def test_is_tensor2():
    """
    x is list, False
    """
    x = [1, 2, 3]
    check = paddle.is_tensor(x)
    assert check == False


@pytest.mark.api_base_is_tensor_vartype
def test_is_tensor3():
    """
    x is num, False
    """
    x = 8
    check = paddle.is_tensor(x)
    assert check == False
