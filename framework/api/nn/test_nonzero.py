#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test nonzero
"""
from apibase import APIBase
from apibase import randtool
import paddle
import pytest
import numpy as np
import numpy.testing as npt


class TestNonzero(APIBase):
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


obj = TestNonzero(paddle.nonzero)


@pytest.mark.api_base_nonzero_vartype
def test_nonzero_base():
    """
    nonzero base
    """
    x = np.array([[1., 1., 4.],
                  [0., 2., 0.],
                  [0., 0., 3.]]).astype(np.float32)
    as_tuple_ = False
    res = np.array([[0, 0],
                    [0, 1],
                    [0, 2],
                    [1, 1],
                    [2, 2]]).astype(np.int64)
    obj.base(res=res, x=x, as_tuple=as_tuple_)


@pytest.mark.api_base_nonzero_parameters
def test_nonzero1():
    """
    x.shape = (3, 3)
    as_tuple = False
    """
    x = np.array([[1., 1., 4.],
                  [0., 2., 0.],
                  [0., 0., 3.]]).astype(np.float32)
    as_tuple_ = False
    res = np.array([[0, 0],
                    [0, 1],
                    [0, 2],
                    [1, 1],
                    [2, 2]]).astype(np.int64)
    obj.run(res=res, x=x, as_tuple=as_tuple_)


@pytest.mark.api_base_nonzero_parameters
def test_nonzero2():
    """
    x.shape = (3, 3)
    as_tuple = True
    """
    paddle.disable_static()
    x = paddle.to_tensor(np.array([[1., 1., 4.],
                                   [0., 2., 0.],
                                   [0., 0., 3.]]).astype(np.float32))
    as_tuple_ = True
    outputs = paddle.nonzero(x, as_tuple_)
    res = np.array([[[0], [0], [0], [1], [2]],
                    [[0], [1], [2], [1], [2]]]).astype(np.int64)
    for i in range(outputs.__len__()):
        out = outputs[i].numpy()
        npt.assert_allclose(out, res[i, :, :])


@pytest.mark.api_base_nonzero_parameters
def test_nonzero3():
    """
    x.shape = (4, )
    as_tuple = False
    """
    x = np.array([2, 1, 0, 3]).astype(np.int32)
    as_tuple_ = False
    res = np.array([[0],
                    [1],
                    [3]]).astype(np.int64)
    obj.run(res=res, x=x, as_tuple=as_tuple_)


@pytest.mark.api_base_nonzero_parameters
def test_nonzero4():
    """
    x.shape = (4, )
    as_tuple = True
    """
    paddle.disable_static()
    x = paddle.to_tensor(np.array([2, 1, 0, 3]).astype(np.int32))
    as_tuple_ = True
    outputs = paddle.nonzero(x, as_tuple_)
    res = np.array([[[0],
                     [1],
                     [3]]]).astype(np.int64)
    for i in range(outputs.__len__()):
        out = outputs[i].numpy()
        npt.assert_allclose(out, res[i, :])


@pytest.mark.api_base_nonzero_parameters
def test_nonzero5():
    """
    x.shape = (3, 2, 2, 2)
    as_tuple = False
    """
    x = np.array([[[[1., 2.],
                    [0., 0.]],

                   [[5., 0.],
                    [7., 8.]]],


                  [[[9., 10.],
                    [0., 12.]],

                   [[13., 0.],
                    [15., 16.]]],


                  [[[17., 0.],
                    [0., 0.]],

                   [[0., 22.],
                    [23., 24.]]]]).astype(np.float64)
    as_tuple_ = False
    res = np.array([[0, 0, 0, 0],
                    [0, 0, 0, 1],
                    [0, 1, 0, 0],
                    [0, 1, 1, 0],
                    [0, 1, 1, 1],
                    [1, 0, 0, 0],
                    [1, 0, 0, 1],
                    [1, 0, 1, 1],
                    [1, 1, 0, 0],
                    [1, 1, 1, 0],
                    [1, 1, 1, 1],
                    [2, 0, 0, 0],
                    [2, 1, 0, 1],
                    [2, 1, 1, 0],
                    [2, 1, 1, 1]]).astype(np.int64)
    obj.run(res=res, x=x, as_tuple=as_tuple_)


@pytest.mark.api_base_nonzero_parameters
def test_nonzero6():
    """
    x.shape = (3, 2, 2, 2)
    as_tuple = True
    """
    paddle.disable_static()
    x = np.array([[[[1., 2.],
                    [0., 0.]],

                   [[5., 0.],
                    [7., 8.]]],


                  [[[9., 10.],
                    [0., 12.]],

                   [[13., 0.],
                    [15., 16.]]],


                  [[[17., 0.],
                    [0., 0.]],

                   [[0., 22.],
                    [23., 24.]]]]).astype(np.float64)
    x = paddle.to_tensor(x)
    as_tuple_ = True
    outputs = paddle.nonzero(x, as_tuple_)
    res = np.array([[[0.], [0.], [0.], [0.], [0.], [1.], [1.],
                     [1.], [1.], [1.], [1.], [2.], [2.], [2.], [2.]],
                    [[0.], [0.], [1.], [1.], [1.], [0.], [0.],
                     [0.], [1.], [1.], [1.], [0.], [1.], [1.], [1.]],
                    [[0.], [0.], [0.], [1.], [1.], [0.], [0.],
                     [1.], [0.], [1.], [1.], [0.], [0.], [1.], [1.]],
                    [[0.], [1.], [0.], [0.], [1.], [0.], [1.],
                     [1.], [0.], [0.], [1.], [0.], [1.], [0.], [1.]]]).astype(np.int64)
    for i in range(outputs.__len__()):
        out = outputs[i].numpy()
        npt.assert_allclose(out, res[i, :, :])
