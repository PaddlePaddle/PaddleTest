#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_searchsorted
"""
from apibase import APIBase
import paddle
import pytest
import numpy as np


class TestSearchSorted(APIBase):
    """
    test
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32, np.float64, np.int32, np.int64]
        # self.debug = True
        # self.static = True
        # enable check grad
        self.enable_backward = False


obj = TestSearchSorted(paddle.searchsorted)


@pytest.mark.api_base_searchsorted_vartype
def test_selu_base():
    """
    base
    """
    sorted_sequence = np.array([1, 3, 4, 7, 9])
    values = np.array([4, 8])
    res = np.array([2, 4])
    obj.base(res=res, sorted_sequence=sorted_sequence, values=values)


@pytest.mark.api_base_searchsorted_parameters
def test_selu0():
    """
    default
    sorted_sequence: vector
    values: vector
    """
    sorted_sequence = np.array([1, 3, 4, 7, 9])
    values = np.array([1, 4, 8])
    res = np.array([0, 2, 4])
    obj.base(res=res, sorted_sequence=sorted_sequence, values=values)


@pytest.mark.api_base_searchsorted_parameters
def test_selu1():
    """
    default
    sorted_sequence: tensor-2D
    values: tensor-2D
    """
    sorted_sequence = np.array([[1, 3, 4, 7, 9], [2, 7, 8, 10, 20]])
    values = np.array([[1, 4, 8], [2, 10, 18]])
    res = np.array([[0, 2, 4], [0, 3, 4]])
    obj.base(res=res, sorted_sequence=sorted_sequence, values=values)


@pytest.mark.api_base_searchsorted_parameters
def test_selu2():
    """
    default
    sorted_sequence: tensor-1D
    values: tensor-2D
    """
    sorted_sequence = np.array([1, 3, 4, 7, 9])
    values = np.array([[1, 4, 8], [2, 7, 8]])
    res = np.array([[0, 2, 4], [1, 3, 4]])
    obj.base(res=res, sorted_sequence=sorted_sequence, values=values)


@pytest.mark.api_base_searchsorted_parameters
def test_selu3():
    """
    default
    sorted_sequence: tensor-1D
    values: tensor-2D
    """
    sorted_sequence = np.array([1, 3, 4, 7, 9])
    values = np.array([[1, 4, 8], [2, 7, 8]])
    res = np.array([[0, 2, 4], [1, 3, 4]])
    obj.base(res=res, sorted_sequence=sorted_sequence, values=values)


@pytest.mark.api_base_searchsorted_parameters
def test_selu4():
    """
    default
    sorted_sequence: tensor-1D
    values: tensor-2D
    """
    sorted_sequence = np.array([1, 3, 4, 7, 9])
    values = np.array([[1, 4, 8], [2, 7, 8]])
    res = np.array([[0, 2, 4], [1, 3, 4]])
    obj.base(res=res, sorted_sequence=sorted_sequence, values=values)


@pytest.mark.api_base_searchsorted_parameters
def test_selu5():
    """
    default
    sorted_sequence: tensor-1D
    values: tensor-2D
    """
    sorted_sequence = np.array([1, 3, 4, 7, 9])
    values = np.array([[1, 4], [2, 7], [1, 9], [8, 9]])
    res = np.array([[0, 2], [1, 3], [0, 4], [4, 4]])
    obj.base(res=res, sorted_sequence=sorted_sequence, values=values)


@pytest.mark.api_base_searchsorted_parameters
def test_selu5():
    """
    default
    sorted_sequence: tensor-1D
    values: tensor-2D, out of range
    """
    sorted_sequence = np.array([1, 3, 4, 7, 9])
    values = np.array([[-1, -4], [-2, 17], [11, 91], [-8, 19]])
    res = np.array([[0, 0], [0, 5], [5, 5], [0, 5]])
    obj.base(res=res, sorted_sequence=sorted_sequence, values=values)


@pytest.mark.api_base_searchsorted_parameters
def test_selu6():
    """
    default
    sorted_sequence: tensor-1D
    values: tensor-2D
    right = True
    """
    sorted_sequence = np.array([1, 3, 4, 7, 9, 11, 15])
    values = np.array([[1, 4], [2, 7], [1, 14], [8, 9]])
    res = np.array([[1, 3], [1, 4], [1, 6], [4, 5]])
    obj.base(res=res, sorted_sequence=sorted_sequence, values=values, right=True)


@pytest.mark.api_base_searchsorted_parameters
def test_selu6():
    """
    default
    sorted_sequence: tensor-1D
    values: tensor-3D
    right = True
    """
    sorted_sequence = np.array([1, 3, 4, 7, 9, 11, 15])
    values = np.array([[[1, 4], [2, 7]], [[1, 14], [8, 9]]])
    res = np.array([[[1, 3], [1, 4]], [[1, 6], [4, 5]]])
    obj.base(res=res, sorted_sequence=sorted_sequence, values=values, right=True)
