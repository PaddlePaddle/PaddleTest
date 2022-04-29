#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test nansum
"""
from apibase import APIBase
from apibase import randtool
from apibase import compare
import paddle
import pytest
import numpy as np


@pytest.mark.api_base_nansum_parameters
def test_nansum0():
    """
    base
    """
    x = randtool("float", -10, 10, (3, 3, 3))
    x[-1, :, :] = float("nan")
    x[:, 1, :] = float("-nan")
    x = x.astype("float32")
    res = paddle.to_tensor(x).nansum()
    exp = [np.nansum(x)]
    compare(res.numpy(), exp)


@pytest.mark.api_base_nansum_parameters
def test_nansum():
    """
    x = randtool("float", -10, 10, (3, 3, 3))
    x[-1, :, :] = float('nan')
    x[:, 1, :] = float('-nan')
    """
    x = randtool("float", -10, 10, (3, 3, 3))
    x[-1, :, :] = float("nan")
    x[:, 1, :] = float("-nan")
    x = x.astype("float64")
    res = paddle.to_tensor(x).nansum()
    exp = [np.nansum(x)]
    compare(res.numpy(), exp)


@pytest.mark.api_base_nansum_parameters
def test_nansum1():
    """
    x = randtool("float", -10, 10, (3, 3, 3))
    x[-1, :, :] = float('nan')
    x[:, 1, :] = float('-nan')
    res = np.nansum(x, axis=-1)
    """
    x = randtool("float", -10, 10, (3, 3, 3))
    x[-1, :, :] = float("nan")
    x[:, 1, :] = float("-nan")
    res = paddle.to_tensor(x).nansum(axis=-1)
    exp = np.nansum(x, axis=-1)
    compare(res.numpy(), exp)


@pytest.mark.api_base_nansum_parameters
def test_nansum2():
    """
    x = randtool("float", -10, 10, (3, 3, 3))
    x[-1, :, :] = float('nan')
    x[:, 1, :] = float('-nan')
    res = np.nansum(x, axis=0, keepdim=True)
    """
    x = randtool("float", -10, 10, (3, 3, 3))
    x[-1, :, :] = float("nan")
    x[:, 1, :] = float("-nan")
    res = paddle.to_tensor(x).nansum(axis=0, keepdim=True)
    exp = np.nansum(x, axis=0, keepdims=True)
    compare(res.numpy(), exp)


@pytest.mark.api_base_nansum_parameters
def test_nansum3():
    """
    x=0
    """
    x = randtool("float", -10, 10, (3, 2, 3, 4, 5, 1, 2))
    x[-1, :, 2, 2, :, :, :] = float("nan")
    x[:, 1, :, :, :, :, -1] = float("-nan")
    res = paddle.to_tensor(x).nansum(axis=3, keepdim=True)
    exp = np.nansum(x, axis=3, keepdims=True)
    compare(res.numpy(), exp)
