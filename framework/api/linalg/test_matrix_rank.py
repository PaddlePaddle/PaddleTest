#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test matrix_rank
"""
from apibase import APIBase
import paddle
import pytest
import numpy as np


class TestMatrixRank(APIBase):
    """
    test matrix_rank
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32, np.float64]
        # self.debug = True
        # enable check grad
        self.enable_backward = False


obj = TestMatrixRank(paddle.linalg.matrix_rank)


@pytest.mark.api_linalg_matrix_rank_vartype
def test_matrix_rank_base():
    """
    base
    """
    tol = None
    hermitian = False
    x = np.array(
        [[-2.0, 2.0, 1.0, 2.0, 2.0], [-2.0, 2.0, 1.0, 2.0, 2.0], [1.0, 3.0, 4.0, 1.0, 7.0], [-4.0, 4.0, 2.0, 4.0, 4.0]]
    )
    res = [np.linalg.matrix_rank(x, tol=tol, hermitian=hermitian)]
    obj.base(res=res, x=x, tol=tol, hermitian=hermitian)


@pytest.mark.api_linalg_matrix_rank_parameters
def test_matrix_rank_1():
    """
    x.shape=[4, 5]
    tol = 4.4
    hermitian = False
    """
    tol = 4.4
    hermitian = False
    x = np.array(
        [[-2.0, 2.0, 1.0, 2.0, 2.0], [-2.0, 2.0, 1.0, 2.0, 2.0], [1.0, 3.0, 4.0, 1.0, 7.0], [-4.0, 4.0, 2.0, 4.0, 4.0]]
    )
    res = [np.linalg.matrix_rank(x, tol=tol, hermitian=hermitian)]
    obj.run(res=res, x=x, tol=tol, hermitian=hermitian)


@pytest.mark.api_linalg_matrix_rank_parameters
def test_matrix_rank_2():
    """
    x.shape=[4, 4]
    tol = 4.4
    hermitian = True
    """
    tol = 4.4
    hermitian = True
    x = np.array([[-2.0, 2.0, 1.0, 2.0], [-2.0, 2.0, 1.0, 2.0], [1.0, 3.0, 4.0, 1.0], [-4.0, 4.0, 2.0, 4.0]])
    res = [np.linalg.matrix_rank(x, tol=tol, hermitian=hermitian)]
    obj.run(res=res, x=x, tol=tol, hermitian=hermitian)


@pytest.mark.api_linalg_matrix_rank_parameters
def test_matrix_rank_3():
    """
    x.shape=[4, 4]
    tol = paddle.to_tensor(np.array([4.4]).astype(np.float32))
    hermitian = True
    """
    tol_n = np.array([4.4]).astype(np.float32)
    hermitian = True
    x = np.array([[-2.0, 2.0, 1.0, 2.0], [-2.0, 2.0, 1.0, 2.0], [1.0, 3.0, 4.0, 1.0], [-4.0, 4.0, 2.0, 4.0]])
    res = np.linalg.matrix_rank(x, tol=tol_n, hermitian=hermitian)
    obj.dtype = np.float32
    obj.run(res=res, x=x, tol=tol_n, hermitian=hermitian)
    obj.dtype = None


@pytest.mark.api_linalg_matrix_rank_parameters
def test_matrix_rank_4():
    """
    x.shape=[2, 3, 4, 4]
    tol = paddle.to_tensor(np.array([4.4]).astype(np.float32))
    hermitian = True
    """
    tol_n = np.array([4.4]).astype(np.float32)
    hermitian = True
    x = np.array(
        [
            [
                [[-2.0, 2.0, 1.0, 2.0], [-2.0, 2.0, 1.0, 2.0], [1.0, 3.0, 4.0, 1.0], [-4.0, 4.0, 2.0, 4.0]],
                [[-2.0, 2.0, 1.0, 2.0], [-2.0, 2.0, 1.0, 2.0], [1.0, 3.0, 4.0, 1.0], [-4.0, 4.0, 2.0, 4.0]],
                [[-2.0, 2.0, 1.0, 2.0], [-2.0, 2.0, 1.0, 2.0], [1.0, 3.0, 4.0, 1.0], [-4.0, 4.0, 2.0, 4.0]],
            ],
            [
                [[-2.0, 2.0, 1.0, 2.0], [-2.0, 2.0, 1.0, 2.0], [1.0, 3.0, 4.0, 1.0], [-4.0, 4.0, 2.0, 4.0]],
                [[-2.0, 2.0, 1.0, 2.0], [-2.0, 2.0, 1.0, 2.0], [1.0, 3.0, 4.0, 1.0], [-4.0, 4.0, 2.0, 4.0]],
                [[-2.0, 2.0, 1.0, 2.0], [-2.0, 2.0, 1.0, 2.0], [1.0, 3.0, 4.0, 1.0], [-4.0, 4.0, 2.0, 4.0]],
            ],
        ]
    )
    res = np.linalg.matrix_rank(x, tol=tol_n, hermitian=hermitian)
    obj.dtype = np.float32
    obj.run(res=res, x=x, tol=tol_n, hermitian=hermitian)
    obj.dtype = None


@pytest.mark.api_linalg_matrix_rank_parameters
def test_matrix_rank_5():
    """
    x.shape=[2, 3, 4, 4]
    tol = paddle.to_tensor(np.array([[4.4, 4.5, 4.4],[4.4, 4.5, 4.4]]).astype(np.float32))
    hermitian = True
    """
    tol_n = np.array([[4.4, 4.5, 4.4], [4.4, 4.5, 4.4]]).astype(np.float32)
    hermitian = True
    x = np.array(
        [
            [
                [[-2.0, 2.0, 1.0, 2.0], [-2.0, 2.0, 1.0, 2.0], [1.0, 3.0, 4.0, 1.0], [-4.0, 4.0, 2.0, 4.0]],
                [[-2.0, 2.0, 1.0, 2.0], [-2.0, 2.0, 1.0, 2.0], [1.0, 3.0, 4.0, 1.0], [-4.0, 4.0, 2.0, 4.0]],
                [[-2.0, 2.0, 1.0, 2.0], [-2.0, 2.0, 1.0, 2.0], [1.0, 3.0, 4.0, 1.0], [-4.0, 4.0, 2.0, 4.0]],
            ],
            [
                [[-2.0, 2.0, 1.0, 2.0], [-2.0, 2.0, 1.0, 2.0], [1.0, 3.0, 4.0, 1.0], [-4.0, 4.0, 2.0, 4.0]],
                [[-2.0, 2.0, 1.0, 2.0], [-2.0, 2.0, 1.0, 2.0], [1.0, 3.0, 4.0, 1.0], [-4.0, 4.0, 2.0, 4.0]],
                [[-2.0, 2.0, 1.0, 2.0], [-2.0, 2.0, 1.0, 2.0], [1.0, 3.0, 4.0, 1.0], [-4.0, 4.0, 2.0, 4.0]],
            ],
        ]
    )
    res = np.linalg.matrix_rank(x, tol=tol_n, hermitian=hermitian)
    obj.dtype = np.float32
    obj.run(res=res, x=x, tol=tol_n, hermitian=hermitian)
    obj.dtype = None


@pytest.mark.api_linalg_matrix_rank_parameters
def test_matrix_rank_6():
    """
    x.shape=[2, 2, 4, 4]
    tol = paddle.to_tensor(np.array([[4.4]]).astype(np.float32))
    hermitian = True
    """
    tol_n = np.array([[4.4]]).astype(np.float32)
    hermitian = True
    x = np.array(
        [
            [
                [[-2.0, 2.0, 1.0, 2.0], [-2.0, 2.0, 1.0, 2.0], [1.0, 3.0, 4.0, 1.0], [-4.0, 4.0, 2.0, 4.0]],
                [[-2.0, 2.0, 1.0, 2.0], [-2.0, 2.0, 1.0, 2.0], [1.0, 3.0, 4.0, 1.0], [-4.0, 4.0, 2.0, 4.0]],
            ],
            [
                [[-2.0, 2.0, 1.0, 2.0], [-2.0, 2.0, 1.0, 2.0], [1.0, 3.0, 4.0, 1.0], [-4.0, 4.0, 2.0, 4.0]],
                [[-2.0, 2.0, 1.0, 2.0], [-2.0, 2.0, 1.0, 2.0], [1.0, 3.0, 4.0, 1.0], [-4.0, 4.0, 2.0, 4.0]],
            ],
        ]
    )
    res = np.linalg.matrix_rank(x, tol=tol_n, hermitian=hermitian)
    obj.dtype = np.float32
    obj.run(res=res, x=x, tol=tol_n, hermitian=hermitian)
    obj.dtype = None


@pytest.mark.api_linalg_matrix_rank_parameters
def test_matrix_rank_7():
    """
    x.shape=[2, 4, 4, 4]
    tol = paddle.to_tensor(np.array([[4.4], [4.5]]).astype(np.float32))
    hermitian = True
    """
    tol_n = np.array([[4.4], [4.5]]).astype(np.float32)
    hermitian = True
    x = np.array(
        [
            [
                [[-2.0, 2.0, 1.0, 2.0], [-2.0, 2.0, 1.0, 2.0], [1.0, 3.0, 4.0, 1.0], [-4.0, 4.0, 2.0, 4.0]],
                [[-2.0, 2.0, 1.0, 2.0], [-2.0, 2.0, 1.0, 2.0], [1.0, 3.0, 4.0, 1.0], [-4.0, 4.0, 2.0, 4.0]],
                [[-2.0, 2.0, 1.0, 2.0], [-2.0, 2.0, 1.0, 2.0], [1.0, 3.0, 4.0, 1.0], [-4.0, 4.0, 2.0, 4.0]],
                [[-2.0, 2.0, 1.0, 2.0], [-2.0, 2.0, 1.0, 2.0], [1.0, 3.0, 4.0, 1.0], [-4.0, 4.0, 2.0, 4.0]],
            ],
            [
                [[-2.0, 2.0, 1.0, 2.0], [-2.0, 2.0, 1.0, 2.0], [1.0, 3.0, 4.0, 1.0], [-4.0, 4.0, 2.0, 4.0]],
                [[-2.0, 2.0, 1.0, 2.0], [-2.0, 2.0, 1.0, 2.0], [1.0, 3.0, 4.0, 1.0], [-4.0, 4.0, 2.0, 4.0]],
                [[-2.0, 2.0, 1.0, 2.0], [-2.0, 2.0, 1.0, 2.0], [1.0, 3.0, 4.0, 1.0], [-4.0, 4.0, 2.0, 4.0]],
                [[-2.0, 2.0, 1.0, 2.0], [-2.0, 2.0, 1.0, 2.0], [1.0, 3.0, 4.0, 1.0], [-4.0, 4.0, 2.0, 4.0]],
            ],
        ]
    )
    res = np.linalg.matrix_rank(x, tol=tol_n, hermitian=hermitian)
    obj.dtype = np.float32
    obj.run(res=res, x=x, tol=tol_n, hermitian=hermitian)
    obj.dtype = None


@pytest.mark.api_linalg_matrix_rank_parameters
def test_matrix_rank_8():
    """
    x.shape=[2, 4, 4, 5]
    tol = paddle.to_tensor(np.array([[4.4], [4.5]]).astype(np.float32))
    hermitian = False
    """
    tol_n = np.array([[4.4], [4.5]]).astype(np.float32)
    hermitian = False
    x = np.array(
        [
            [
                [
                    [-2.0, 2.0, 1.0, 2.0, 4.0],
                    [-2.0, 2.0, 1.0, 2.0, 4.0],
                    [1.0, 3.0, 4.0, 1.0, 7.0],
                    [-4.0, 4.0, 2.0, 4.0, 8.0],
                ],
                [
                    [-2.0, 2.0, 1.0, 2.0, 4.0],
                    [-2.0, 2.0, 1.0, 2.0, 4.0],
                    [1.0, 3.0, 4.0, 1.0, 7.0],
                    [-4.0, 4.0, 2.0, 4.0, 8.0],
                ],
                [
                    [-2.0, 2.0, 1.0, 2.0, 4.0],
                    [-2.0, 2.0, 1.0, 2.0, 4.0],
                    [1.0, 3.0, 4.0, 1.0, 7.0],
                    [-4.0, 4.0, 2.0, 4.0, 8],
                ],
                [
                    [-2.0, 2.0, 1.0, 2.0, 4.0],
                    [-2.0, 2.0, 1.0, 2.0, 4.0],
                    [1.0, 3.0, 4.0, 1.0, 7.0],
                    [-4.0, 4.0, 2.0, 4.0, 8],
                ],
            ],
            [
                [
                    [-2.0, 2.0, 1.0, 2.0, 4.0],
                    [-2.0, 2.0, 1.0, 2.0, 4.0],
                    [1.0, 3.0, 4.0, 1.0, 7.0],
                    [-4.0, 4.0, 2.0, 4.0, 8],
                ],
                [
                    [-2.0, 2.0, 1.0, 2.0, 4.0],
                    [-2.0, 2.0, 1.0, 2.0, 4.0],
                    [1.0, 3.0, 4.0, 1.0, 7.0],
                    [-4.0, 4.0, 2.0, 4.0, 8],
                ],
                [
                    [-2.0, 2.0, 1.0, 2.0, 4.0],
                    [-2.0, 2.0, 1.0, 2.0, 4.0],
                    [1.0, 3.0, 4.0, 1.0, 7.0],
                    [-4.0, 4.0, 2.0, 4.0, 8],
                ],
                [
                    [-2.0, 2.0, 1.0, 2.0, 4.0],
                    [-2.0, 2.0, 1.0, 2.0, 4.0],
                    [1.0, 3.0, 4.0, 1.0, 7.0],
                    [-4.0, 4.0, 2.0, 4.0, 8],
                ],
            ],
        ]
    )
    res = np.linalg.matrix_rank(x, tol=tol_n, hermitian=hermitian)
    obj.dtype = np.float32
    obj.run(res=res, x=x, tol=tol_n, hermitian=hermitian)
    obj.dtype = None


@pytest.mark.api_linalg_matrix_rank_parameters
def test_matrix_rank_9():
    """
    x.shape=[2, 4, 4, 4]
    tol = paddle.to_tensor(np.array([[4.4], [4.5]]).astype(np.float32))
    hermitian = True
    """
    tol_n = np.array([[4.4], [4.5]]).astype(np.float32)
    hermitian = True
    x = np.array(
        [
            [
                [[-2.0, 2.0, 1.0, 2.0], [-2.0, 2.0, 1.0, 2.0], [1.0, 3.0, 4.0, 1.0], [-4.0, 4.0, 2.0, 4.0]],
                [[-2.0, 2.0, 1.0, 2.0], [-2.0, 2.0, 1.0, 2.0], [1.0, 3.0, 4.0, 1.0], [-4.0, 4.0, 2.0, 4.0]],
                [[-2.0, 2.0, 1.0, 2.0], [-2.0, 2.0, 1.0, 2.0], [1.0, 3.0, 4.0, 1.0], [-4.0, 4.0, -2.0, 4.0]],
                [[-2.0, 2.0, 1.0, 2.0], [-2.0, 2.0, 1.0, 2.0], [1.0, 3.0, 4.0, 1.0], [-4.0, 4.0, 2.0, 4.0]],
            ],
            [
                [[-2.0, 2.0, 1.0, 2.0], [-2.0, 3.0, 1.0, 2.0], [1.0, 3.0, 4.0, 1.0], [-4.0, 4.0, 2.0, 4.0]],
                [[-2.0, 2.0, 1.0, 2.0], [-2.0, 2.0, 1.0, 2.0], [1.0, 3.0, 4.0, 1.0], [-4.0, 4.0, 2.0, 4.0]],
                [[-2.0, 2.0, 1.0, 2.0], [-2.0, 2.0, 1.0, 2.0], [1.0, 3.0, 4.0, 1.0], [-4.0, 4.0, -2.0, 4.0]],
                [[-2.0, 2.0, 1.0, 2.0], [-2.0, 2.0, 1.0, 2.0], [1.0, 3.0, 4.0, 1.0], [-4.0, 4.0, 2.0, 4.0]],
            ],
        ]
    )
    res = np.linalg.matrix_rank(x, tol=tol_n, hermitian=hermitian)
    obj.dtype = np.float32
    obj.run(res=res, x=x, tol=tol_n, hermitian=hermitian)
    obj.dtype = None


@pytest.mark.api_linalg_matrix_rank_parameters
def test_matrix_rank_11():
    """
    x.shape=[2, 4, 4, 4]
    tol = None
    hermitian = True
    """
    tol_n = None
    hermitian = True
    x = np.array(
        [
            [
                [[-2.0, 2.0, 1.0, 2.0], [-2.0, 2.0, 1.0, 2.0], [1.0, 3.0, 4.0, 1.0], [-4.0, 4.0, 2.0, 4.0]],
                [[-2.0, 2.0, 1.0, 2.0], [-2.0, 2.0, 1.0, 2.0], [1.0, 3.0, 4.0, 1.0], [-4.0, 4.0, 2.0, 4.0]],
                [[-2.0, 2.0, 1.0, 2.0], [-2.0, 2.0, 1.0, 2.0], [1.0, 3.0, 4.0, 1.0], [-4.0, 4.0, -2.0, 4.0]],
                [[-2.0, 2.0, 1.0, 2.0], [-2.0, 2.0, 1.0, 2.0], [1.0, 3.0, 4.0, 1.0], [-4.0, 4.0, 2.0, 4.0]],
            ],
            [
                [[-2.0, 2.0, 1.0, 2.0], [-2.0, 3.0, 1.0, 2.0], [1.0, 3.0, 4.0, 1.0], [-4.0, 4.0, 2.0, 4.0]],
                [[-2.0, 2.0, 1.0, 2.0], [-2.0, 2.0, 1.0, 2.0], [1.0, 3.0, 4.0, 1.0], [-4.0, 4.0, 2.0, 4.0]],
                [[-2.0, 2.0, 1.0, 2.0], [-2.0, 2.0, 1.0, 2.0], [1.0, 3.0, 4.0, 1.0], [-4.0, 4.0, -2.0, 4.0]],
                [[-2.0, 2.0, 1.0, 2.0], [-2.0, 2.0, 1.0, 2.0], [1.0, 3.0, 4.0, 1.0], [-4.0, 4.0, 2.0, 4.0]],
            ],
        ]
    )
    res = np.linalg.matrix_rank(x, tol=tol_n, hermitian=hermitian)
    obj.dtype = np.float32
    obj.run(res=res, x=x, tol=tol_n, hermitian=hermitian)
    obj.dtype = None
