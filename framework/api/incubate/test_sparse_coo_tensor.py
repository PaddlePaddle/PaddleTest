#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test sparse_coo_tensor
"""

import sys
import paddle
from paddle.fluid.framework import _test_eager_guard
import pytest
import numpy as np

sys.path.append("../../utils/")
from interceptor import skip_branch_not_develop


# @skip_branch_not_develop
@pytest.mark.api_sparse_sparse_coo_tensor_vartype
def test_sparse_sparse_coo_tensor_base():
    """
    base
    """
    types = ["float32", "float64"]
    with _test_eager_guard():
        indices = [[0, 1, 2], [1, 2, 0]]
        values = [1.0, 2.0, 3.0]
        dense_shape = [3, 3]
        for dtype in types:
            coo = paddle.incubate.sparse.sparse_coo_tensor(indices, values, dense_shape, dtype=dtype)
            dense = coo.to_dense()
            dense_res = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 2.0], [3.0, 0.0, 0.0]])

            assert np.allclose(coo.indices().numpy(), np.array(indices))
            assert np.allclose(coo.values().numpy(), np.array(values))
            assert np.allclose(dense.numpy(), dense_res)


# @skip_branch_not_develop
@pytest.mark.api_sparse_sparse_coo_tensor_vartype
def test_sparse_sparse_coo_tensor1():
    """
    3d tensor
    """
    types = ["float32", "float64"]
    with _test_eager_guard():
        indices = [[0, 1, 2], [1, 2, 0], [1, 2, 0]]
        values = [1.0, 2.0, 3.0]
        dense_shape = [3, 3, 3]
        for dtype in types:
            coo = paddle.incubate.sparse.sparse_coo_tensor(indices, values, dense_shape, dtype=dtype)
            dense = coo.to_dense()
            dense_res = np.zeros((3, 3, 3))
            for i in range(3):
                dense_res[indices[0][i]][indices[1][i]][indices[2][i]] = values[i]

            assert np.allclose(coo.indices().numpy(), np.array(indices))
            assert np.allclose(coo.values().numpy(), np.array(values))
            assert np.allclose(dense.numpy(), dense_res)
