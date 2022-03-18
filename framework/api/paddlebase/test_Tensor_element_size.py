#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_Tensor_element_size
"""

import paddle
import pytest


@pytest.mark.api_base_element_size_parameters
def test_element_size0():
    """
    bool
    """
    a = paddle.to_tensor(1, dtype="bool")
    num = paddle.Tensor.element_size(a)
    assert num == 1


@pytest.mark.api_base_element_size_parameters
def test_element_size1():
    """
    uint8
    """
    a = paddle.to_tensor(1, dtype="uint8")
    num = paddle.Tensor.element_size(a)
    assert num == 1


@pytest.mark.api_base_element_size_parameters
def test_element_size2():
    """
    uint16
    """
    a = paddle.to_tensor(1, dtype="uint16")
    num = paddle.Tensor.element_size(a)
    assert num == 2


@pytest.mark.api_base_element_size_parameters
def test_element_size3():
    """
    int8
    """
    a = paddle.to_tensor(1, dtype="int8")
    num = paddle.Tensor.element_size(a)
    assert num == 1


@pytest.mark.api_base_element_size_parameters
def test_element_size4():
    """
    int16
    """
    a = paddle.to_tensor(1, dtype="int16")
    num = paddle.Tensor.element_size(a)
    assert num == 2


@pytest.mark.api_base_element_size_parameters
def test_element_size5():
    """
    int32
    """
    a = paddle.to_tensor(1, dtype="int32")
    num = paddle.Tensor.element_size(a)
    assert num == 4


@pytest.mark.api_base_element_size_parameters
def test_element_size6():
    """
    int64
    """
    a = paddle.to_tensor(1, dtype="int64")
    num = paddle.Tensor.element_size(a)
    assert num == 8


@pytest.mark.api_base_element_size_parameters
def test_element_size7():
    """
    float16
    """
    a = paddle.to_tensor(1, dtype="float16")
    num = paddle.Tensor.element_size(a)
    assert num == 2


@pytest.mark.api_base_element_size_parameters
def test_element_size8():
    """
    float32
    """
    a = paddle.to_tensor(1, dtype="float32")
    num = paddle.Tensor.element_size(a)
    assert num == 4


@pytest.mark.api_base_element_size_parameters
def test_element_size9():
    """
    float64
    """
    a = paddle.to_tensor(1, dtype="float64")
    num = paddle.Tensor.element_size(a)
    assert num == 8


@pytest.mark.api_base_element_size_parameters
def test_element_size10():
    """
    complex64
    """
    a = paddle.to_tensor(1, dtype="complex64")
    num = paddle.Tensor.element_size(a)
    assert num == 8


@pytest.mark.api_base_element_size_parameters
def test_element_size11():
    """
    complex128
    """
    a = paddle.to_tensor(1, dtype="complex128")
    num = paddle.Tensor.element_size(a)
    assert num == 16
