#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test kron
"""
from apibase import APIBase
from apibase import randtool
import paddle
import pytest
import numpy as np


class TestKron(APIBase):
    """
    test broadcast
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.int32, np.int64, np.float32, np.float64]
        # self.debug = True
        # self.static = True
        # enable check grad
        self.enable_backward = True


obj = TestKron(paddle.kron)


@pytest.mark.api_base_kron_vartype
def test_kron_base():
    """
    x.shape: (2, 3)
    y.shape: (3, 3)
    """
    x = np.array([[1.0, 2.0, 1.0], [1.0, 2.0, 3.0]]).astype(np.float32)
    y = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]).astype(np.float32)
    res = np.array(
        [
            [1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0, 14, 16.0, 18.0, 7.0, 8.0, 9.0],
            [1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 3.0, 6.0, 9.0],
            [4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 12.0, 15.0, 18.0],
            [7.0, 8.0, 9.0, 14.0, 16.0, 18.0, 21.0, 24.0, 27.0],
        ]
    ).astype(np.float32)
    obj.base(res=res, x=x, y=y)


@pytest.mark.api_base_kron_parameters
def test_kron1():
    """
    x.shape: (2, 3)
    y.shape: (3, 3)
    """
    x = np.array([[1.0, 2.0, 1.0], [1.0, 2.0, 3.0]]).astype(np.float32)
    y = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]).astype(np.float32)
    res = np.array(
        [
            [1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0, 14, 16.0, 18.0, 7.0, 8.0, 9.0],
            [1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 3.0, 6.0, 9.0],
            [4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 12.0, 15.0, 18.0],
            [7.0, 8.0, 9.0, 14.0, 16.0, 18.0, 21.0, 24.0, 27.0],
        ]
    ).astype(np.float32)
    obj.run(res=res, x=x, y=y)


@pytest.mark.api_base_kron_parameters
def test_kron2():
    """
    x.shape: (2, 3)
    y.shape: (3, 3)
    """
    x = np.array([[1, 2], [3, 4]]).astype(np.int64)
    y = np.array([[[1, 2], [4, 5], [7, 8]], [[1, 2], [4, 5], [7, 8]], [[1, 2], [4, 5], [7, 8]]]).astype(np.int64)
    res = np.array(
        [
            [[1, 2, 2, 4], [4, 5, 8, 10], [7, 8, 14, 16], [3, 6, 4, 8], [12, 15, 16, 20], [21, 24, 28, 32]],
            [[1, 2, 2, 4], [4, 5, 8, 10], [7, 8, 14, 16], [3, 6, 4, 8], [12, 15, 16, 20], [21, 24, 28, 32]],
            [[1, 2, 2, 4], [4, 5, 8, 10], [7, 8, 14, 16], [3, 6, 4, 8], [12, 15, 16, 20], [21, 24, 28, 32]],
        ]
    ).astype(np.int64)
    obj.run(res=res, x=x, y=y)


@pytest.mark.api_base_kron_parameters
def test_kron3():
    """
    x.shape: (2, 2)
    y.shape: (3, 3)
    """
    x = np.array([[1.5, 2.5], [3.5, 4.5]]).astype(np.float64)
    y = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]).astype(np.float64)
    res = np.array(
        [
            [1.50000000, 3.0, 4.50000000, 2.50000000, 5.0, 7.50000000],
            [6.0, 7.50000000, 9.0, 10.0, 12.50000000, 15.0],
            [10.50000000, 12.0, 13.50000000, 17.50000000, 20.0, 22.50000000],
            [3.50000000, 7.0, 10.50000000, 4.50000000, 9.0, 13.50000000],
            [14.0, 17.50000000, 21.0, 18.0, 22.50000000, 27.0],
            [24.50000000, 28.0, 31.50000000, 31.50000000, 36.0, 40.50000000],
        ]
    ).astype(np.float64)
    obj.run(res=res, x=x, y=y)


@pytest.mark.api_base_kron_parameters
def test_kron4():
    """
    x.shape: (1)
    y.shape: (3, 3)
    """
    x = np.array([2]).astype(np.int64)
    y = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).astype(np.int64)
    res = np.array([[2, 4, 6], [8, 10, 12], [14, 16, 18]]).astype(np.int64)
    obj.run(res=res, x=x, y=y)
