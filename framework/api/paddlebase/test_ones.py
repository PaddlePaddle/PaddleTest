#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test ones
"""
from apibase import APIBase
import paddle
import pytest
import numpy as np

is_in_eager = paddle.fluid.framework._in_eager_without_dygraph_check()


class TestOnes(APIBase):
    """
    test ones
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.int32, np.int64]
        # self.debug = True
        # enable check grad
        self.enable_backward = False


obj = TestOnes(paddle.ones)


# test dtype=np.*
@pytest.mark.api_base_ones_vartype
def test_ones_base():
    """
    shape_type=list, dtype=np.float32
    """
    shape = [2, 4]
    dtype = np.float32
    res = np.ones(shape, dtype=dtype)
    obj.base(res=res, shape=shape, dtype=dtype)


@pytest.mark.api_base_ones_parameters
def test_ones():
    """
    shape_type=tuple
    """
    shape = (2, 4, 1)
    res = np.ones(shape)
    obj.run(res=res, shape=shape)


@pytest.mark.api_base_ones_parameters
def test_ones1():
    """
    shape_type=tuple, dtype=np.int32
    """
    shape = (2, 4)
    dtype = np.int32
    res = np.ones(shape, dtype=dtype)
    obj.run(res=res, shape=shape, dtype=dtype)


@pytest.mark.api_base_ones_parameters
def test_ones2():
    """
    no shape_type=(np.int32)Tensor, dtype=np.int64
    """
    shape = np.array([1]).astype(np.int32)
    dtype = np.int64
    res = np.ones(shape, dtype=dtype)
    obj.run(res=res, shape=shape, dtype=dtype)


@pytest.mark.api_base_ones_parameters
def test_ones3():
    """
    shape_type=list, dtype=np.float16
    """
    shape = [2]
    dtype = np.float16
    res = np.ones(shape, dtype=dtype)
    obj.run(res=res, shape=shape, dtype=dtype)


@pytest.mark.api_base_ones_parameters
def test_ones4():
    """
    shape_type=tuple, dtype=np.float64
    """
    shape = (2, 4, 1)
    dtype = np.float64
    res = np.ones(shape, dtype=dtype)
    obj.run(res=res, shape=shape, dtype=dtype)


# test dtype=bool
@pytest.mark.api_base_ones_parameters
def test_ones5():
    """
    shape_type=list,dtype='bool'
    """
    shape = [1, 2, 3]
    dtype = "bool"
    res = np.ones(shape, dtype=dtype)
    obj.run(res=res, shape=shape, dtype=dtype)


@pytest.mark.api_base_ones_parameters
def test_ones6():
    """
    shape_type=list,dtype=bool
    """
    shape = [1, 2, 3, 4]
    dtype = bool
    res = np.ones(shape, dtype=dtype)
    obj.run(res=res, shape=shape, dtype=dtype)


@pytest.mark.api_base_ones_parameters
def test_ones7():
    """
    shape_type=list,dtype=np.bool
    """
    shape = [1, 5, 2, 3]
    dtype = np.bool
    res = np.ones(shape, dtype=dtype)
    obj.run(res=res, shape=shape, dtype=dtype)


@pytest.mark.api_base_ones_parameters
def test_ones8():
    """
    shape_type=list,dtype=None
    """
    shape = [1, 5, 2, 3]
    dtype = None
    res = np.ones(shape, dtype=dtype)
    obj.run(res=res, shape=shape, dtype=dtype)


# test dtype=str
@pytest.mark.api_base_ones_parameters
def test_ones9():
    """
    shape_type=tuple,dtype='float32'
    """
    shape = (1,)
    dtype = "float32"
    res = np.ones(shape, dtype=dtype)
    obj.run(res=res, shape=shape, dtype=dtype)


@pytest.mark.api_base_ones_parameters
def test_ones10():
    """
    shape_type=list,dtype='float16'
    """
    shape = [5, 2]
    dtype = "float16"
    res = np.ones(shape, dtype=dtype)
    obj.run(res=res, shape=shape, dtype=dtype)


@pytest.mark.api_base_ones_parameters
def test_ones11():
    """
    shape_type=list,dtype='float64'
    """
    shape = [1, 2, 3, 4, 5]
    dtype = "float64"
    res = np.ones(shape, dtype=dtype)
    obj.run(res=res, shape=shape, dtype=dtype)


@pytest.mark.api_base_ones_parameters
def test_ones12():
    """
    shape_type=(np.int64)Tensor,dtype='int32'
    """
    shape = np.array([1, 2]).astype(np.int64)
    dtype = "int32"
    res = np.ones(shape, dtype=dtype)
    obj.run(res=res, shape=shape, dtype=dtype)


@pytest.mark.api_base_ones_parameters
def test_ones13():
    """
    shape_type=tuple,dtype='int64'
    """
    shape = (1, 5, (1))
    dtype = "int64"
    res = np.ones(shape, dtype=dtype)
    obj.run(res=res, shape=shape, dtype=dtype)


# test shape=(np)Tensor
@pytest.mark.api_base_ones_parameters
def test_ones14():
    """
    shape_type=(np)Tensor,shape=np.array([1])
    """
    shape = np.array([1]).astype(np.int64)
    res = np.ones(shape)
    obj.run(res=res, shape=shape)


@pytest.mark.api_base_ones_parameters
def test_ones15():
    """
    shape_type=(np)Tensor,shape=np.array([1, ])
    """
    shape = np.array([1])
    res = np.ones(shape)
    obj.run(res=res, shape=shape)


@pytest.mark.api_base_ones_parameters
def test_ones16():
    """
    shape_type=(np)Tensor,shape=np.array([])
    """
    shape = np.array([]).astype(np.int64)
    res = np.ones(shape)
    obj.run(res=res, shape=shape)


@pytest.mark.api_base_ones_parameters
def test_ones17():
    """
    shape_type=(np)Tensor,shape=np.array([1, (1)])
    """
    shape = np.array([1, (1)]).astype(np.int64)
    res = np.ones(shape)
    obj.run(res=res, shape=shape)


# test shape=list
@pytest.mark.api_base_ones_parameters
def test_ones18():
    """
    shape_type=list,shape=[1000]
    """
    shape = [1000]
    res = np.ones(shape)
    obj.run(res=res, shape=shape)


@pytest.mark.api_base_ones_parameters
def test_ones19():
    """
    shape_type=list,shape=[1, (1)]
    """
    shape = [2, (1)]
    res = np.ones(shape)
    obj.run(res=res, shape=shape)


# test name
@pytest.mark.api_base_ones_parameters
def test_ones20():
    """
    shape_type=list,name=None
    """
    shape = [2]
    name = None
    res = np.ones(shape)
    obj.run(res=res, shape=shape, name=name)


@pytest.mark.api_base_ones_parameters
def test_ones21():
    """
    shape_type=list,name='None'
    """
    shape = [2]
    name = "None"
    res = np.ones(shape)
    obj.run(res=res, shape=shape, name=name)


@pytest.mark.api_base_ones_parameters
def test_ones22():
    """
    shape_type=list,name="ones_name"
    """
    shape = [2]
    name = "ones_name"
    res = np.ones(shape)
    obj.run(res=res, shape=shape, name=name)


# test shape_type=tuple
@pytest.mark.api_base_ones_parameters
def test_ones23():
    """
    shape_type=tuple,shape=(1,)
    """
    shape = (1,)
    res = np.ones(shape)
    obj.run(res=res, shape=shape)


# def test_ones24():
#     """
#     shape_type=list,shape=[],static no,icafe,static no,icafe,todo:fix RC
#     """
#     shape = []
#     res = np.ones(shape)
#     obj.run(res=res, shape=shape)


# def test_ones25():
#     """
#     shape_type=tuple,shape=(),static no,icafe,static no,icafe,todo:fix RC
#     """
#     shape = ()
#     res = np.ones(shape)
#     obj.run(res=res, shape=shape)


# def test_ones26():
#     """
#     shape_type=(np)Tensor,shape=np.array([0])->[],None,icafe
#     """
#     shape = np.array([0])
#     res = np.ones(shape)
#     obj.run(res=res, shape=shape)


# def test_ones27():
#     """
#     shape_type=list,shape=[0]->[],None,icafe
#     """
#     shape = (0)
#     res = np.ones(shape)
#     obj.run(res=res, shape=shape)


# def test_ones28():
#     """
#     shape_type=list,shape=[0, ]->[],None,icafe
#     """
#     shape = [0, ]
#     res = np.ones(shape)
#     obj.run(res=res, shape=shape)


# exception case
@pytest.mark.api_base_ones_exception
def test_ones29():
    """
    shape_type=list,shape_value=[2, [3]],static TypeError
    """
    shape = [2, [3]]
    obj.exception(mode="c", etype="InvalidArgumentError", shape=shape)


@pytest.mark.api_base_ones_exception
def test_ones30():
    """
    no shape_type=tuple,shape=(0),AttributeError
    """
    shape = 0
    etype = ValueError if is_in_eager else AttributeError
    obj.exception(mode="python", etype=etype, shape=shape)


@pytest.mark.api_base_ones_exception
def test_ones31():
    """
    shape_type=tuple,shape=(1.1),AttributeError
    """
    shape = 1.1
    etype = ValueError if is_in_eager else AttributeError
    obj.exception(mode="python", etype=etype, shape=shape)


@pytest.mark.api_base_ones_exception
def test_ones32():
    """
    shape_type=tuple,shape=(1000, 1),dtype=np.int8,static TypeError
    """
    shape = (1000, 1)
    dtype = np.int8
    obj.exception(mode="c", etype="NotFoundError", shape=shape, dtype=dtype)


@pytest.mark.api_base_ones_exception
def test_ones33():
    """
    shape_type=list,dtype='BOOL'
    """
    shape = [1, 2, 3, 4]
    dtype = "BOOL"
    obj.exception(mode="python", etype=TypeError, shape=shape, dtype=dtype)


@pytest.mark.api_base_ones_exception
def test_ones34():
    """
    shape_type=list,shape=-1,c++ error,static TypeError
    """
    shape = [-1, 5]
    obj.exception(mode="c", etype="InvalidArgumentError", shape=shape)


# def test_ones35():
#     """
#     shape_type=list,shape=[1.1],static TypeError
#     """
#     shape = [1.1, 2.1]
#     obj.exception(mode='python', etype=TypeError, shape=shape)
