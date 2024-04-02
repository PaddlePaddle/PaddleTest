#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test diagflat
"""
from apibase import APIBase
import paddle
import pytest
import numpy as np


class TestDiagFlat(APIBase):
    """
    test nonzero
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.int32, np.int64, np.float32, np.float64]
        # self.debug = True
        # self.static = True
        # enable check grad
        self.enable_backward = False


obj = TestDiagFlat(paddle.diagflat)


@pytest.mark.api_base_diagflat_vartype
def test_diagflat():
    """
    diagflat base
    """
    x = np.array([1.0, 2.0, 3.0, 4.0])
    offset = 0
    res = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0], [0.0, 0.0, 3.0, 0.0], [0.0, 0.0, 0.0, 4.0]])
    obj.base(res=res, x=x, offset=offset)


@pytest.mark.api_base_diagflat_parameters
def test_diagflat1():
    """
    x=np.array([1.0, 2.0, 3.0, 4.0])
    offset = 1
    """
    x = np.array([1.0, 2.0, 3.0, 4.0]).astype(np.float32)
    offset = 1
    res = np.array(
        [
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 3.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 4.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    ).astype(np.float32)
    obj.run(res=res, x=x, offset=offset)


@pytest.mark.api_base_diagflat_parameters
def test_diagflat2():
    """
    x=np.array([1.0, 2.0, 3.0, 4.0])
    offset = -1
    """
    x = np.array([1.0, 2.0, 3.0, 4.0]).astype(np.float32)
    offset = -1
    res = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 3.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 4.0, 0.0],
        ]
    ).astype(np.float32)
    obj.run(res=res, x=x, offset=offset)


@pytest.mark.api_base_diagflat_parameters
def test_diagflat3():
    """
    diagflat base
    """
    x = np.arange(1, 97).reshape(3, 2, 2, 1, 2, 4).astype(np.float32)
    offset = -1
    res = np.diagflat(x, k=offset).astype(np.float32)
    obj.run(res=res, x=x, offset=offset)


@pytest.mark.api_base_diagflat_parameters
def test_diagflat4():
    """
    diagflat base
    """
    x = np.arange(1, 3 * 2 * 2 * 2 * 4 * 2 * 2 + 1).reshape(3, 2, 2, 1, 2, 4, 2, 2).astype(np.float32)
    offset = 2
    res = np.diagflat(x, k=offset).astype(np.float32)
    obj.run(res=res, x=x, offset=offset)
