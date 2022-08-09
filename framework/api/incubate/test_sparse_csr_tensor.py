#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test sparse_csr_tensor
"""

import sys
import paddle
from paddle.fluid.framework import _test_eager_guard
import pytest
import numpy as np

sys.path.append("../../utils/")
from interceptor import skip_branch_not_develop


@skip_branch_not_develop
@pytest.mark.api_sparse_csr_tensor_vartype
def test_sparse_sparse_csr_tensor_base():
    """
    base
    """
    types = ["float32", "float64"]
    with _test_eager_guard():
        crows = [0, 2, 3, 5]
        cols = [1, 3, 2, 0, 1]
        values = [1, 2, 3, 4, 5]
        dense_shape = [3, 4]
        for dtype in types:
            csr = paddle.incubate.sparse.sparse_csr_tensor(crows, cols, values, dense_shape, dtype=dtype)
            dense = csr.to_dense()
            dense_res = np.array([[0, 1, 0, 2], [0, 0, 3, 0], [4, 5, 0, 0]])

            assert np.allclose(csr.crows().numpy(), np.array(crows))
            assert np.allclose(csr.cols().numpy(), np.array(cols))
            assert np.allclose(dense.numpy(), dense_res)


@skip_branch_not_develop
@pytest.mark.api_sparse_csr_tensor_parameters
def test_sparse_sparse_csr_tensor1():
    """
    every row have only one none-0 element
    """
    types = ["float32", "float64"]
    with _test_eager_guard():
        crows = [0, 1, 2, 3, 4]
        cols = [0, 1, 2, 3]
        values = [1, 2, 3, 4]
        dense_shape = [4, 4]
        for dtype in types:
            csr = paddle.incubate.sparse.sparse_csr_tensor(crows, cols, values, dense_shape, dtype=dtype)
            dense = csr.to_dense()
            dense_res = np.diag([1, 2, 3, 4])

            assert np.allclose(csr.crows().numpy(), np.array(crows))
            assert np.allclose(csr.cols().numpy(), np.array(cols))
            assert np.allclose(dense.numpy(), dense_res)


@skip_branch_not_develop
@pytest.mark.api_sparse_csr_tensor_parameters
def test_sparse_csr_tensor2():
    """
    3d tensor
    """
    types = ["float32", "float64"]
    with _test_eager_guard():
        crows = [0, 1, 2, 3, 4]
        cols = [0, 1, 2, 3]
        values = [1, 2, 3, 4]
        dense_shape = [4, 4]
        for dtype in types:
            csr = paddle.incubate.sparse.sparse_csr_tensor(crows, cols, values, dense_shape, dtype=dtype)
            dense = csr.to_dense()
            dense_res = np.diag([1, 2, 3, 4])

            assert np.allclose(csr.crows().numpy(), np.array(crows))
            assert np.allclose(csr.cols().numpy(), np.array(cols))
            assert np.allclose(dense.numpy(), dense_res)
