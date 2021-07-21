#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test eye
"""
from apibase import APIBase
import paddle
import pytest
import numpy as np


class TestEye(APIBase):
    """
    test eye
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float16, np.float32, np.float64, np.int32, np.int64]
        self.enable_backward = False


obj = TestEye(paddle.eye)


@pytest.mark.api_base_eye_vartype
def test_eye_base():
    """
    base
    """
    num_rows = 3
    num_columns = 5
    res = np.eye(num_rows, num_columns)
    obj.base(res=res, num_rows=num_rows, num_columns=num_columns)


@pytest.mark.api_base_eye_parameters
def test_eye1():
    """
    dtype use default value
    """
    num_rows = 3
    res = np.eye(num_rows)
    obj.run(res=res, num_rows=num_rows)


@pytest.mark.api_base_eye_parameters
def test_eye2():
    """
    dtype='float16'
    """
    num_rows = 3
    dtype = 'float16'
    res = np.eye(num_rows)
    obj.run(res=res, num_rows=num_rows, dtype=dtype)


@pytest.mark.api_base_eye_parameters
def test_eye3():
    """
    dtype='float32'
    """
    num_rows = 5
    num_columns = 3
    dtype = 'float32'
    res = np.eye(num_rows, num_columns)
    obj.run(res=res, num_rows=num_rows, num_columns=num_columns, dtype=dtype)


@pytest.mark.api_base_eye_parameters
def test_eye4():
    """
    dtype='float64'
    """
    num_rows = 5
    num_columns = 3
    dtype = 'float64'
    res = np.eye(num_rows, num_columns)
    obj.run(res=res, num_rows=num_rows, num_columns=num_columns, dtype=dtype)


# def test_eye5():
#     """
#     dtype='int32'
#     this case will fail because when num_rows=0, dygraph returns [] and static returns none(it's a bug)
#     """
#     num_rows = 0
#     dtype = 'int32'
#     res = np.eye(num_rows)
#     obj.run(res=res, num_rows=num_rows, dtype=dtype)


@pytest.mark.api_base_eye_parameters
def test_eye6():
    """
    dtype='int64'
    """
    num_rows = 5
    dtype = 'int64'
    res = np.eye(num_rows)
    obj.run(res=res, num_rows=num_rows, dtype=dtype)


@pytest.mark.api_base_eye_parameters
def test_eye7():
    """
    TypeError:dtype=bool
    """
    num_rows = 5
    dtype = 'bool'
    obj.exception(mode='c', etype='NotFoundError', num_rows=num_rows, dtype=dtype)


@pytest.mark.api_base_eye_parameters
def test_eye8():
    """
    TypeError:num_rows=negtive_num
    """
    num_rows = -3
    obj.exception(mode='python', etype=TypeError, num_rows=num_rows, dtype=np.float16)


@pytest.mark.api_base_eye_parameters
def test_eye9():
    """
    TypeError:num_rows=float
    """
    num_rows = 3.4
    obj.exception(mode='python', etype=TypeError, num_rows=num_rows)


@pytest.mark.api_base_eye_parameters
def test_eye10():
    """
    dtype=np.dtype
    """
    num_rows = 5
    dtype = np.float16
    res = np.eye(num_rows)
    obj.run(res=res, num_rows=num_rows, dtype=dtype)
