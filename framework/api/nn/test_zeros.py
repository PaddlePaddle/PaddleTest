#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test zeros
"""
from apibase import APIBase
import paddle
import pytest
import numpy as np


class TestZeros(APIBase):
    """
    test zeros
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.int32, np.int64]
        # self.debug = True
        # enable check grad
        self.enable_backward = False


obj = TestZeros(paddle.zeros)


# test dtype=np.*
@pytest.mark.api_base_zeros_vartype
def test_zeros_base():
    """
    shape_type=list, dtype=np.float32
    """
    shape = [2, 4]
    dtype = np.float32
    res = np.zeros(shape, dtype=dtype)
    obj.base(res=res, shape=shape, dtype=dtype)


@pytest.mark.api_base_zeros_parameters
def test_zeros():
    """
    shape_type=tuple
    """
    shape = (2, 4, 1)
    res = np.zeros(shape)
    obj.run(res=res, shape=shape)


@pytest.mark.api_base_zeros_parameters
def test_zeros1():
    """
    shape_type=tuple, dtype=np.int32
    """
    shape = (2, 4)
    dtype = np.int32
    res = np.zeros(shape, dtype=dtype)
    obj.run(res=res, shape=shape, dtype=dtype)


@pytest.mark.api_base_zeros_parameters
def test_zeros2():
    """
    no shape_type=(np.int32)Tensor, dtype=np.int64
    """
    shape = np.array([1]).astype(np.int32)
    dtype = np.int64
    res = np.zeros(shape, dtype=dtype)
    obj.run(res=res, shape=shape, dtype=dtype)


@pytest.mark.api_base_zeros_parameters
def test_zeros3():
    """
    shape_type=list, dtype=np.float16
    """
    shape = [2]
    dtype = np.float16
    res = np.zeros(shape, dtype=dtype)
    obj.run(res=res, shape=shape, dtype=dtype)


@pytest.mark.api_base_zeros_parameters
def test_zeros4():
    """
    shape_type=tuple, dtype=np.float64
    """
    shape = (2, 4, 1)
    dtype = np.float64
    res = np.zeros(shape, dtype=dtype)
    obj.run(res=res, shape=shape, dtype=dtype)


# test dtype=bool
@pytest.mark.api_base_zeros_parameters
def test_zeros5():
    """
    shape_type=list,dtype='bool'
    """
    shape = [1, 2, 3]
    dtype = 'bool'
    res = np.zeros(shape, dtype=dtype)
    obj.run(res=res, shape=shape, dtype=dtype)


@pytest.mark.api_base_zeros_parameters
def test_zeros6():
    """
    shape_type=list,dtype=bool
    """
    shape = [1, 2, 3, 4]
    dtype = bool
    res = np.zeros(shape, dtype=dtype)
    obj.run(res=res, shape=shape, dtype=dtype)


@pytest.mark.api_base_zeros_parameters
def test_zeros7():
    """
    shape_type=list,dtype=np.bool
    """
    shape = [1, 5, 2, 3]
    dtype = np.bool
    res = np.zeros(shape, dtype=dtype)
    obj.run(res=res, shape=shape, dtype=dtype)


@pytest.mark.api_base_zeros_parameters
def test_zeros8():
    """
    shape_type=list,dtype=None
    """
    shape = [1, 5, 2, 3]
    dtype = None
    res = np.zeros(shape, dtype=dtype)
    obj.run(res=res, shape=shape, dtype=dtype)


# test dtype=str
@pytest.mark.api_base_zeros_parameters
def test_zeros9():
    """
    shape_type=tuple,dtype='float32'
    """
    shape = (1,)
    dtype = 'float32'
    res = np.zeros(shape, dtype=dtype)
    obj.run(res=res, shape=shape, dtype=dtype)


@pytest.mark.api_base_zeros_parameters
def test_zeros10():
    """
    shape_type=list,dtype='float16'
    """
    shape = [5, 2]
    dtype = 'float16'
    res = np.zeros(shape, dtype=dtype)
    obj.run(res=res, shape=shape, dtype=dtype)


@pytest.mark.api_base_zeros_parameters
def test_zeros11():
    """
    shape_type=list,dtype='float64'
    """
    shape = [1, 2, 3, 4, 5]
    dtype = 'float64'
    res = np.zeros(shape, dtype=dtype)
    obj.run(res=res, shape=shape, dtype=dtype)


@pytest.mark.api_base_zeros_parameters
def test_zeros12():
    """
    shape_type=(np.int64)Tensor,dtype='int32'
    """
    shape = np.array([1, 2]).astype(np.int64)
    dtype = 'int32'
    res = np.zeros(shape, dtype=dtype)
    obj.run(res=res, shape=shape, dtype=dtype)


@pytest.mark.api_base_zeros_parameters
def test_zeros13():
    """
    shape_type=tuple,dtype='int64'
    """
    shape = (1, 5, (1))
    dtype = 'int64'
    res = np.zeros(shape, dtype=dtype)
    obj.run(res=res, shape=shape, dtype=dtype)


# test shape=(np)Tensor
@pytest.mark.api_base_zeros_parameters
def test_zeros14():
    """
    shape_type=(np)Tensor,shape=np.array([1])
    """
    shape = np.array([1]).astype(np.int64)
    res = np.zeros(shape)
    obj.run(res=res, shape=shape)


@pytest.mark.api_base_zeros_parameters
def test_zeros15():
    """
    shape_type=(np)Tensor,shape=np.array([1, ])
    """
    shape = np.array([1, ])
    res = np.zeros(shape)
    obj.run(res=res, shape=shape)


@pytest.mark.api_base_zeros_parameters
def test_zeros16():
    """
    shape_type=(np)Tensor,shape=np.array([])
    """
    shape = np.array([]).astype(np.int64)
    res = np.zeros(shape)
    obj.run(res=res, shape=shape)


@pytest.mark.api_base_zeros_parameters
def test_zeros17():
    """
    shape_type=(np)Tensor,shape=np.array([1, (1)])
    """
    shape = np.array([1, (1)]).astype(np.int64)
    res = np.zeros(shape)
    obj.run(res=res, shape=shape)


# test shape=list
@pytest.mark.api_base_zeros_parameters
def test_zeros18():
    """
    shape_type=list,shape=[1000]
    """
    shape = [1000]
    res = np.zeros(shape)
    obj.run(res=res, shape=shape)


@pytest.mark.api_base_zeros_parameters
def test_zeros19():
    """
    shape_type=list,shape=[1, (1)]
    """
    shape = [2, (1)]
    res = np.zeros(shape)
    obj.run(res=res, shape=shape)


# test name
@pytest.mark.api_base_zeros_parameters
def test_zeros20():
    """
    shape_type=list,name=None
    """
    shape = [2, ]
    name = None
    res = np.zeros(shape)
    obj.run(res=res, shape=shape, name=name)


@pytest.mark.api_base_zeros_parameters
def test_zeros21():
    """
    shape_type=list,name='None'
    """
    shape = [2, ]
    name = 'None'
    res = np.zeros(shape)
    obj.run(res=res, shape=shape, name=name)


@pytest.mark.api_base_zeros_parameters
def test_zeros22():
    """
    shape_type=list,name="zeros_name"
    """
    shape = [2, ]
    name = "zeros_name"
    res = np.zeros(shape)
    obj.run(res=res, shape=shape, name=name)


# test shape_type=tuple
@pytest.mark.api_base_zeros_parameters
def test_zeros23():
    """
    shape_type=tuple,shape=(1,)
    """
    shape = (1,)
    res = np.zeros(shape)
    obj.run(res=res, shape=shape)


# def test_zeros24():
#     """
#     shape_type=list,shape=[],static no,icafe,todo:fix RC
#     """
#     shape = []
#     res = np.zeros(shape)
#     obj.run(res=res, shape=shape)


# def test_zeros25():
#     """
#     shape_type=tuple,shape=(),static no,icafe,todo:fix RC
#     """
#     shape = ()
#     res = np.zeros(shape)
#     obj.run(res=res, shape=shape)


# def test_zeros26():
#     """
#     shape_type=(np)Tensor,shape=np.array([0])->[],None,icafe
#     """
#     shape = np.array([0])
#     res = np.zeros(shape)
#     obj.run(res=res, shape=shape)


# def test_zeros27():
#     """
#     shape_type=list,shape=[0]->[],None,icafe
#     """
#     shape = (0)
#     res = np.zeros(shape)
#     obj.run(res=res, shape=shape)


# def test_zeros28():
#     """
#     shape_type=list,shape=[0, ]->[],None,icafe
#     """
#     shape = [0, ]
#     res = np.zeros(shape)
#     obj.run(res=res, shape=shape)


# exception case
@pytest.mark.api_base_zeros_exception
def test_zeros29():
    """
    shape_type=list,shape_value=[2, [3]],static TypeError
    """
    shape = [2, [3]]
    obj.exception(mode='c', etype='InvalidArgumentError', shape=shape)


@pytest.mark.api_base_zeros_exception
def test_zeros30():
    """
    no shape_type=tuple,shape=(0),AttributeError
    """
    shape = (0)
    obj.exception(mode='python', etype=AttributeError, shape=shape)


@pytest.mark.api_base_zeros_exception
def test_zeros31():
    """
    shape_type=tuple,shape=(1.1),AttributeError
    """
    shape = (1.1)
    obj.exception(mode='python', etype=AttributeError, shape=shape)


@pytest.mark.api_base_zeros_exception
def test_zeros32():
    """
    shape_type=tuple,shape=(1000, 1),dtype=np.int8,static TypeError
    """
    shape = (1000, 1)
    dtype = np.int8
    obj.exception(mode='c', etype='NotFoundError', shape=shape, dtype=dtype)


@pytest.mark.api_base_zeros_exception
def test_zeros33():
    """
    shape_type=list,dtype='BOOL'
    """
    shape = [1, 2, 3, 4]
    dtype = 'BOOL'
    obj.exception(mode='python', etype=TypeError, shape=shape, dtype=dtype)


@pytest.mark.api_base_zeros_exception
def test_zeros34():
    """
    shape_type=list,shape=-1,static TypeError
    """
    shape = [-1, 5]
    obj.exception(mode='c', etype='InvalidArgumentError', shape=shape)


# def test_zeros35():
#     """
#     shape=[1.1],static TypeError
#     """
#     shape = [1.1, 2.1]
#     obj.exception(mode='python', etype=TypeError, shape=shape)
