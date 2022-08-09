#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test renorm
"""
from apibase import APIBase
from apibase import randtool
import paddle
import pytest
import numpy as np


class TestRenorm(APIBase):
    """
    test renorm
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32, np.float64]
        self.static = False
        # self.debug = True
        # enable check grad
        # self.enable_backward = False


obj = TestRenorm(paddle.renorm)


@pytest.mark.api_base_renorm_vartype
def test_renorm_base():
    """
    base
    """
    p = 1
    axis = 0
    max_norm = 5
    x = np.array(
        [[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], [[1.5, 1.5, 1.5], [0.0, 0.0, 0.0]], [[2.5, 2.5, 2.5], [3.0, 3.0, 3.0]]]
    )
    res = np.array(
        [
            [[0.55555556, 0.55555556, 0.55555556], [1.11111111, 1.11111111, 1.11111111]],
            [[1.50000000, 1.50000000, 1.50000000], [0.0, 0.0, 0.0]],
            [[0.75757576, 0.75757576, 0.75757576], [0.90909091, 0.90909091, 0.90909091]],
        ]
    )
    obj.base(res=res, x=x, p=p, axis=axis, max_norm=max_norm)


@pytest.mark.api_base_renorm_parameters
def test_renorm():
    """
    p = 2
    axis = 1
    max_norm = 40
    """
    p = 2
    axis = 1
    max_norm = 40
    x = np.array(
        [[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], [[10.0, 10.0, 10.0], [0.0, 0.0, 0.0]], [[6.0, 6.0, 6.0], [3.0, 3.0, 3.0]]]
    )
    res = np.array(
        [[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], [[10.0, 10.0, 10.0], [0.0, 0.0, 0.0]], [[6.0, 6.0, 6.0], [3.0, 3.0, 3.0]]]
    )
    obj.run(res=res, x=x, p=p, axis=axis, max_norm=max_norm)


@pytest.mark.api_base_renorm_parameters
def test_renorm1():
    """
    p = 2
    axis = 1
    max_norm = 20
    """
    p = 2
    axis = 1
    max_norm = 20
    x = np.array(
        [[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], [[10.0, 10.0, 10.0], [0.0, 0.0, 0.0]], [[6.0, 6.0, 6.0], [3.0, 3.0, 3.0]]]
    )
    res = np.array(
        [
            [[0.98652725, 0.98652725, 0.98652725], [2.0, 2.0, 2.0]],
            [[9.86527247, 9.86527247, 9.86527247], [0.0, 0.0, 0.0]],
            [[5.91916348, 5.91916348, 5.91916348], [3.0, 3.0, 3.0]],
        ]
    )

    obj.run(res=res, x=x, p=p, axis=axis, max_norm=max_norm)


@pytest.mark.api_base_renorm_parameters
def test_renorm2():
    """
    p = 2
    axis = 1
    max_norm = 50
    """
    p = 2
    axis = 1
    max_norm = 50
    x = np.array(
        [[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], [[10.0, 10.0, 10.0], [0.0, 0.0, 0.0]], [[6.0, 6.0, 6.0], [3.0, 3.0, 3.0]]]
    )
    res = np.array(
        [[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], [[10.0, 10.0, 10.0], [0.0, 0.0, 0.0]], [[6.0, 6.0, 6.0], [3.0, 3.0, 3.0]]]
    )

    obj.run(res=res, x=x, p=p, axis=axis, max_norm=max_norm)


@pytest.mark.api_base_renorm_parameters
def test_renorm3():
    """
    p = 1.5
    axis = 2
    max_norm = 6
    """
    p = 1.5
    axis = 2
    max_norm = 20
    x = np.array(
        [[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], [[10.0, 10.0, 10.0], [0.0, 0.0, 0.0]], [[6.0, 6.0, 6.0], [3.0, 3.0, 3.0]]]
    )
    res = np.array(
        [[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], [[10.0, 10.0, 10.0], [0.0, 0.0, 0.0]], [[6.0, 6.0, 6.0], [3.0, 3.0, 3.0]]]
    )

    obj.run(res=res, x=x, p=p, axis=axis, max_norm=max_norm)


@pytest.mark.api_base_renorm_parameters
def test_renorm4():
    """
    p = 1.2
    axis = 2
    max_norm = 6.5
    """
    p = 1.2
    axis = 2
    max_norm = 6.5
    x = np.array(
        [[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], [[10.0, 10.0, 10.0], [0.0, 0.0, 0.0]], [[6.0, 6.0, 6.0], [3.0, 3.0, 3.0]]]
    )
    res = np.array(
        [
            [[0.36700647, 0.36700647, 0.36700647], [0.73401295, 0.73401295, 0.73401295]],
            [[3.67006474, 3.67006474, 3.67006474], [0.0, 0.0, 0.0]],
            [[2.20203885, 2.20203885, 2.20203885], [1.10101942, 1.10101942, 1.10101942]],
        ]
    )

    obj.run(res=res, x=x, p=p, axis=axis, max_norm=max_norm)
