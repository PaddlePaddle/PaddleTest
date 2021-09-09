#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_chunk.py
"""
from apibase import APIBase
import paddle
import pytest
import numpy as np


class TestChunk(APIBase):
    """
    test chunk
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.int32, np.int64, np.float16, np.float32, np.float64]
        self.enable_backward = False
        # static_graph of chunk api is not supported in this frame
        self.static = False
        self.dygraph = True
        self.debug = True
        self.no_grad_var = ["axis"]


obj = TestChunk(paddle.chunk)


@pytest.mark.api_base_chunk_vartype
def test_chunk_base():
    """
    base
    """
    x = np.arange(9).reshape(3, 3)
    chunks = 3
    axis = 0
    res = np.split(x, 3, 0)
    obj.base(res=res, x=x, chunks=chunks, axis=axis)


@pytest.mark.api_base_chunk_parameters
def test_chunk1():
    """
    axis=Tensor
    """
    x = np.arange(9).reshape(3, 3)
    chunks = 3
    axis = np.array([0]).astype("int64")
    res = np.split(x, 3, 0)
    obj.base(res=res, x=x, chunks=chunks, axis=axis)


@pytest.mark.api_base_chunk_parameters
def test_chunk2():
    """
    axis<0
    """
    x = np.arange(9).reshape(3, 3)
    chunks = 3
    res = np.split(x, 3, 1)
    obj.base(res=res, x=x, chunks=chunks, axis=-1)


class TestChunk1(APIBase):
    """
    test chunk
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.bool]
        self.enable_backward = False
        # static_graph of chunk api is not supported in this frame
        self.static = False
        self.dygraph = True
        self.debug = True
        # self.no_grad_var = ["index", "axis"]


obj1 = TestChunk1(paddle.chunk)


@pytest.mark.api_base_chunk_parameters
def test_chunk5():
    """
    base
    """
    x = np.array([1, 0, 1])
    chunks = 3
    axis = 0
    res = [np.array([True]), np.array([False]), np.array([True])]
    obj1.base(res=res, x=x, chunks=chunks, axis=axis)
