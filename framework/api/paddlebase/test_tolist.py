#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test tolist
"""
from apibase import APIBase
import paddle
import pytest
import numpy as np


@pytest.mark.api_base_tolist_vartype
def test_tolist_base():
    """
    base
    """
    x = [0.0, 1.1, 2.1, 1.1, 5.4]
    x_tensor = paddle.to_tensor(x)
    res = paddle.tolist(x_tensor)
    assert isinstance(res, list)
    assert np.allclose(np.array(res), np.array(x))


@pytest.mark.api_base_tolist_parameters
def test_tolist1():
    """
    base
    """
    x = [[0.0, 1.1, 2.1, 1.1, 5.4], [0.0, 1.1, 2.1, 1.1, 5.4]]
    x_tensor = paddle.to_tensor(x)
    res = paddle.tolist(x_tensor)
    assert isinstance(res, list)
    assert np.allclose(np.array(res), np.array(x))
