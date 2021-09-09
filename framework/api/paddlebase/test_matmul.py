#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test paddle.matmul
"""

from apibase import APIBase
import paddle
import pytest
import numpy as np


class TestMatmul(APIBase):
    """
    test matmul
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32, np.float64]
        # self.debug = True
        # self.static = True
        # enable check grad
        self.enable_backward = True


obj = TestMatmul(paddle.matmul)


@pytest.mark.api_base_matmul_vartype
def test_matmul_base():
    """
    matmul_base
    """
    np.random.seed(0)
    x_data = np.random.random([10]).astype(np.float32)
    y_data = np.random.random([10]).astype(np.float32)
    res = np.array([3.58071361])
    obj.base(res=res, x=x_data, y=y_data)


@pytest.mark.api_base_matmul_parameters
def test_matmul_vector_vector():
    """
    vector * vector
    """
    np.random.seed(0)
    x_data = np.random.random([10]).astype(np.float32)
    y_data = np.random.random([10]).astype(np.float32)
    res = np.array([3.58071361])
    obj.run(res=res, x=x_data, y=y_data)


@pytest.mark.api_base_matmul_parameters
def test_matmul_matrix_vector():
    """
    matrix * vector
    """
    np.random.seed(0)
    x_data = np.random.random([10, 5]).astype(np.float32)
    y_data = np.random.random([5]).astype(np.float32)
    res = np.matmul(x_data, y_data)
    obj.run(res=res, x=x_data, y=y_data)


@pytest.mark.api_base_matmul_parameters
def test_matmul_batched_matrix_broadcasted_vector():
    """
    batched_matrix * broadcasted_vector
    """
    np.random.seed(0)
    x_data = np.random.random([10, 5, 2]).astype(np.float32)
    y_data = np.random.random([2]).astype(np.float32)
    res = np.matmul(x_data, y_data)
    obj.run(res=res, x=x_data, y=y_data)


@pytest.mark.api_base_matmul_parameters
def test_matmul_batched_matrix_batched_matrix():
    """
    batched_matrix * batched_matrix
    """
    np.random.seed(0)
    x_data = np.random.random([10, 5, 2]).astype(np.float32)
    y_data = np.random.random([10, 2, 5]).astype(np.float32)
    res = np.matmul(x_data, y_data)
    obj.run(res=res, x=x_data, y=y_data)


@pytest.mark.api_base_matmul_parameters
def test_matmul_batched_matrix_broadcasted_matrix():
    """
    batched_matrix * broadcasted_matrix
    """
    np.random.seed(0)
    x_data = np.random.random([10, 1, 5, 2]).astype(np.float32)
    y_data = np.random.random([1, 3, 2, 5]).astype(np.float32)
    res = np.matmul(x_data, y_data)
    obj.run(res=res, x=x_data, y=y_data)
