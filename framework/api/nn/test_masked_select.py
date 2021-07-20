#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test masked_select
"""
from apibase import APIBase
import paddle
import pytest
import numpy as np


class TestMaskedSelect(APIBase):
    """
    test masked_select
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.int32, np.int64, np.float32, np.float64]
        self.enable_backward = False
        self.no_grad_var = ["mask"]


obj = TestMaskedSelect(paddle.masked_select)


@pytest.mark.api_base_masked_select_vartype
def test_masked_select_base():
    """
    base
    """
    x = np.arange(6).reshape(2, 3)
    mask = np.array([[True, False, False], [False, True, False]])
    res = np.array([0, 4])
    obj.base(res=res, x=x, mask=mask)


@pytest.mark.api_base_masked_select_exception
def test_masked_select1():
    """
    exception:mask is not bool
    """
    x = np.arange(6).reshape(2, 3)
    mask = np.arange(6).reshape(2, 3)
    obj.exception(mode='c', etype='InvalidArgumentError', x=x, mask=mask)


@pytest.mark.api_base_masked_select_exception
def test_masked_select2():
    """
    exception:x type error
    """
    x = [1, 2]
    mask = np.array([True, False])
    obj.exception(mode='c', etype='InvalidArgumentError', x=x, mask=mask)
