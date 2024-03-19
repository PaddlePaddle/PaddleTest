#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test mode
"""
from apibase import APIBase
from apibase import randtool
from apibase import compare
import paddle
import pytest
import numpy as np


@pytest.mark.api_base_mode_vartype
def test_mode_base():
    """
    base
    """
    x = np.array(
        [[[1.0, 1.0, 1.0], [1.0, 2.0, 2.0]], [[1.0, 10.0, 10.0], [1.0, 0.0, 0.0]], [[1.0, 6.0, 6.0], [1.0, 3.0, 3.0]]]
    )
    exp = (np.array([[1.0, 2.0], [10.0, 0.0], [6.0, 3.0]]), np.array([[2, 2], [2, 2], [2, 2]]))
    tmp = paddle.to_tensor(x).mode()
    res = (tmp[0].numpy(), tmp[1].numpy())
    compare(res, exp)


@pytest.mark.api_base_mode_parameters
def test_mode():
    """
    axis = 1
    keepdim = False
    """
    axis = 1
    keepdim = False
    x = np.array(
        [[[1.0, 1.0, 1.0], [1.0, 2.0, 2.0]], [[1.0, 10.0, 10.0], [1.0, 0.0, 0.0]], [[1.0, 6.0, 6.0], [1.0, 3.0, 3.0]]]
    )
    exp = (np.array([[1.0, 1.0, 1.0], [1.0, 0.0, 0.0], [1.0, 3.0, 3.0]]), np.array([[1, 0, 0], [1, 1, 1], [1, 1, 1]]))
    tmp = paddle.to_tensor(x).mode(axis=axis, keepdim=keepdim)
    res = (tmp[0].numpy(), tmp[1].numpy())
    compare(res, exp)


@pytest.mark.api_base_mode_parameters
def test_mode1():
    """
    axis = 2
    keepdim = True
    """
    axis = 2
    keepdim = True
    x = np.array(
        [[[1.0, 1.0, 1.0], [1.0, 2.0, 2.0]], [[1.0, 10.0, 10.0], [1.0, 0.0, 0.0]], [[1.0, 6.0, 6.0], [1.0, 3.0, 3.0]]]
    )
    exp = (np.array([[[1.0], [2.0]], [[10.0], [0.0]], [[6.0], [3.0]]]), np.array([[[2], [2]], [[2], [2]], [[2], [2]]]))
    tmp = paddle.to_tensor(x).mode(axis=axis, keepdim=keepdim)
    res = (tmp[0].numpy(), tmp[1].numpy())
    compare(res, exp)
