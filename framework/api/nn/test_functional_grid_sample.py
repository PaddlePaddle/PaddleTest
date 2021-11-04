#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_functional_grid_sample
"""
import paddle
import paddle.nn.functional as F
import numpy as np
from apibase import compare
import pytest


@pytest.mark.api_nn_grid_sample_parameters
def test_grid_sample_1():
    """
    api: paddle.nn.functional.grid_sample
    x.shape=(2, 3, 128, 128)
    grid.shape=(2, 1000, 1000, 2)
    expect.shape=(2, 3, 1000, 1000)
    """
    x = np.ones((2, 3, 128, 128)).astype("float64")
    grid = np.ones((2, 1000, 1000, 2)).astype("float64")
    x = paddle.to_tensor(x)
    grid = paddle.to_tensor(grid)
    res = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)
    exp = np.ones((2, 3, 1000, 1000)).astype("float64")
    compare(res.numpy(), exp, delta=1e-10, rtol=1e-10)
