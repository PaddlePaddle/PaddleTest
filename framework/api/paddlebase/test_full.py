#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test full
"""
from apibase import APIBase
import paddle
import pytest
import numpy as np


class TestFull(APIBase):
    """
    test paddle.full api
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.bool, np.float16, np.float32, np.float64, np.int32, np.int64]
        # self.debug = True
        # self.static = True
        # enable check grad
        self.no_grad_var = ["shape", "fill_value"]
        self.enable_backward = False


obj = TestFull(paddle.full)


@pytest.mark.api_base_full_vartype
def test_full_base():
    """
    base
    """
    shape = [3, 2]
    fill_value = 1
    res = np.full(shape, fill_value)
    obj.base(res=res, shape=shape, fill_value=fill_value)


@pytest.mark.api_base_full_parameters
def test_full1():
    """
    shape is list, fill_value is bool, bool:True, dtype:bool
    """
    shape = [3, 2]
    fill_value = True
    res = np.full(shape, fill_value)
    obj.run(res=res, shape=shape, fill_value=fill_value, dtype=np.bool)


@pytest.mark.api_base_full_parameters
def test_full2():
    """
    shape is list，fill_value is bool, bool:False, dtype:bool
    """
    shape = [3, 2]
    fill_value = False
    res = np.full(shape, fill_value)
    obj.run(res=res, shape=shape, fill_value=fill_value, dtype=np.bool)


@pytest.mark.api_base_full_parameters
def test_full3():
    """
    shape is list，fill_value is int32, dtype:int32
    """
    shape = [3, 2]
    fill_value = 1
    res = np.full(shape, fill_value).astype(np.int32)
    obj.run(res=res, shape=shape, fill_value=fill_value, dtype=np.int32)


@pytest.mark.api_base_full_parameters
def test_full4():
    """
    shape is list，fill_value is int64, dtype:int64
    """
    shape = [3, 2]
    fill_value = 1
    res = np.full(shape, fill_value)
    obj.run(res=res, shape=shape, fill_value=fill_value, dtype=np.int64)


@pytest.mark.api_base_full_parameters
def test_full5():
    """
    shape is list，fill_value is float16, dtype:float16
    """
    shape = [3, 2]
    fill_value = 1.1
    res = np.full(shape, fill_value).astype(np.float16)
    obj.run(res=res, shape=shape, fill_value=fill_value, dtype=np.float16)


@pytest.mark.api_base_full_parameters
def test_full6():
    """
    shape is list，fill_value is float32, dtype:float32
    """
    shape = [3, 2]
    fill_value = 1.1
    res = np.full(shape, fill_value)
    obj.run(res=res, shape=shape, fill_value=fill_value, dtype=np.float32)


@pytest.mark.api_base_full_parameters
def test_full7():
    """
    shape is list，fill_value is tensor, dtype:float32
    """
    shape = [3, 2]
    fill_value = np.array([1]).astype(np.float32)
    res = np.full(shape, fill_value)
    obj.run(res=res, shape=shape, fill_value=fill_value, dtype=np.float32)


@pytest.mark.api_base_full_parameters
def test_full8():
    """
    shape is list，fill_value is tensor(type:int32)
    if tensor is not float32, must set dtype == fill_value's type
    """
    shape = [3, 2]
    fill_value = np.array([1]).astype(np.int32)
    res = np.full(shape, fill_value)
    obj.run(res=res, shape=shape, fill_value=fill_value, dtype=np.int32)


@pytest.mark.api_base_full_parameters
def test_full9():
    """
    shape is list，fill_value is tensor, dtype:str
    """
    shape = [3, 2]
    fill_value = np.array([1]).astype(np.float32)
    res = np.full(shape, fill_value)
    obj.run(res=res, shape=shape, fill_value=fill_value, dtype="float32")


@pytest.mark.api_base_full_parameters
def test_full10():
    """
    shape is tuple, fill_value is bool, bool:True, dtype:bool
    """
    shape = (3, 2)
    fill_value = True
    res = np.full(shape, fill_value)
    obj.run(res=res, shape=shape, fill_value=fill_value, dtype=np.bool)


@pytest.mark.api_base_full_parameters
def test_full11():
    """
    shape is tuple，fill_value is bool, bool:False, dtype:bool
    """
    shape = (3, 2)
    fill_value = False
    res = np.full(shape, fill_value)
    obj.run(res=res, shape=shape, fill_value=fill_value, dtype=np.bool)


@pytest.mark.api_base_full_parameters
def test_full12():
    """
    shape is tuple，fill_value is int32, dtype:int32
    """
    shape = (3, 2)
    fill_value = 1
    res = np.full(shape, fill_value)
    obj.run(res=res, shape=shape, fill_value=fill_value, dtype=np.int32)


@pytest.mark.api_base_full_parameters
def test_full13():
    """
    shape is tuple，fill_value is int64, dtype:int64
    """
    shape = (3, 2)
    fill_value = 1
    res = np.full(shape, fill_value)
    obj.run(res=res, shape=shape, fill_value=fill_value, dtype=np.int64)


@pytest.mark.api_base_full_parameters
def test_full14():
    """
    shape is tuple，fill_value is float16, dtype:float16
    """
    shape = (3, 2)
    fill_value = 1.1
    res = np.full(shape, fill_value).astype(np.float16)
    obj.run(res=res, shape=shape, fill_value=fill_value, dtype=np.float16)


@pytest.mark.api_base_full_parameters
def test_full15():
    """
    shape is tuple，fill_value is float32, dtype:float32
    """
    shape = (3, 2)
    fill_value = 1.1
    res = np.full(shape, fill_value)
    obj.run(res=res, shape=shape, fill_value=fill_value, dtype=np.float32)


@pytest.mark.api_base_full_parameters
def test_full16():
    """
    shape is tuple，fill_value is tensor, dtype:float32
    """
    shape = (3, 2)
    fill_value = np.array([1]).astype(np.float32)
    res = np.full(shape, fill_value)
    obj.run(res=res, shape=shape, fill_value=fill_value, dtype=np.float32)


@pytest.mark.api_base_full_parameters
def test_full17():
    """
    shape is tuple，fill_value is tensor(type:int32)
    if tensor is not float32, must set dtype == fill_value's type
    """
    shape = (3, 2)
    fill_value = np.array([1]).astype(np.int32)
    res = np.full(shape, fill_value)
    obj.run(res=res, shape=shape, fill_value=fill_value, dtype=np.int32)


@pytest.mark.api_base_full_parameters
def test_full18():
    """
    shape is tuple，fill_value is tensor, dtype:str
    """
    shape = (3, 2)
    fill_value = np.array([1]).astype(np.float32)
    res = np.full(shape, fill_value)
    obj.run(res=res, shape=shape, fill_value=fill_value, dtype="float32")


@pytest.mark.api_base_full_parameters
def test_full19():
    """
    shape is Tensor, fill_value is bool, bool:True, dtype:bool
    """
    shape = np.array([3, 2])
    fill_value = True
    res = np.full(shape, fill_value)
    obj.run(res=res, shape=shape, fill_value=fill_value, dtype=np.bool)


@pytest.mark.api_base_full_parameters
def test_full20():
    """
    shape is Tensor，fill_value is bool, bool:False, dtype:bool
    """
    shape = np.array([3, 2])
    fill_value = False
    res = np.full(shape, fill_value)
    obj.run(res=res, shape=shape, fill_value=fill_value, dtype=np.bool)


@pytest.mark.api_base_full_parameters
def test_full21():
    """
    shape is Tensor，fill_value is int32, dtype:int32
    """
    shape = np.array([3, 2])
    fill_value = 1
    res = np.full(shape, fill_value)
    obj.run(res=res, shape=shape, fill_value=fill_value, dtype=np.int32)


@pytest.mark.api_base_full_parameters
def test_full22():
    """
    shape is Tensor，fill_value is int64, dtype:int64
    """
    shape = np.array([3, 2])
    fill_value = 1
    res = np.full(shape, fill_value)
    obj.run(res=res, shape=shape, fill_value=fill_value, dtype=np.int64)


@pytest.mark.api_base_full_parameters
def test_full23():
    """
    shape is Tensor，fill_value is float16, dtype:float16
    """
    shape = np.array([3, 2])
    fill_value = 1.1
    res = np.full(shape, fill_value).astype(np.float16)
    obj.run(res=res, shape=shape, fill_value=fill_value, dtype=np.float16)


@pytest.mark.api_base_full_parameters
def test_full24():
    """
    shape is Tensor，fill_value is float32, dtype:float32
    """
    shape = np.array([3, 2])
    fill_value = 1.1
    res = np.full(shape, fill_value)
    obj.run(res=res, shape=shape, fill_value=fill_value, dtype=np.float32)


@pytest.mark.api_base_full_parameters
def test_full25():
    """
    shape is Tensor，fill_value is tensor, dtype:float32
    """
    shape = np.array([3, 2])
    fill_value = np.array([1]).astype(np.float32)
    res = np.full(shape, fill_value)
    obj.run(res=res, shape=shape, fill_value=fill_value, dtype=np.float32)


@pytest.mark.api_base_full_parameters
def test_full26():
    """
    shape is Tensor，fill_value is tensor(type:int32)
    if tensor is not float32, must set dtype == fill_value's type
    """
    shape = np.array([3, 2])
    fill_value = np.array([1]).astype(np.int32)
    res = np.full(shape, fill_value)
    obj.run(res=res, shape=shape, fill_value=fill_value, dtype=np.int32)


@pytest.mark.api_base_full_parameters
def test_full27():
    """
    shape is Tensor，fill_value is tensor, dtype:str
    """
    shape = np.array([3, 2])
    fill_value = np.array([1]).astype(np.float32)
    res = np.full(shape, fill_value)
    obj.run(res=res, shape=shape, fill_value=fill_value, dtype="float32")


@pytest.mark.api_base_full_parameters
def test_full28():
    """
    shape is list, fill_value is bool, bool:True, dtype:bool
    """
    shape = [3, 2]
    fill_value = True
    res = np.full(shape, fill_value)
    obj.run(res=res, shape=shape, fill_value=fill_value, dtype=np.bool)


@pytest.mark.api_base_full_parameters
def test_full29():
    """
    shape is list, fill_value is bool, bool:False, dtype:bool
    """
    shape = [3, 2]
    fill_value = False
    res = np.full(shape, fill_value)
    obj.run(res=res, shape=shape, fill_value=fill_value, dtype=np.bool)


# def test_full30():
#    """
#    shape is Tensor，fill_value is tensor(type:int32)
#    if tensor is not float32, must set dtype == fill_value's type
#    """
#    shape = np.array([3, 2]).astype(np.float16)
#    fill_value = np.array([1]).astype(np.int32)
#    res = np.full(shape, fill_value)
#    obj.exception(etype=TypeError, mode="python", shape=shape, fill_value=fill_value, dtype=np.int32)


# now test frame not support fluid.core.VarDesc.VarType.
# def test_full31():
#     """
#     shape is list, fill_value is bool, bool:True, dtype:core.VarDesc.VarType.bool
#     """
#     shape = [3, 2]
#     fill_value = True
#     res = np.full(shape, fill_value)
#     obj.run(res=res, shape=shape, fill_value=fill_value, dtype=fluid.core.VarDesc.VarType.BOOL)
