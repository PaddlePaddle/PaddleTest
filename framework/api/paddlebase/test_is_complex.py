#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_is_complex
"""
import paddle
import pytest


@pytest.mark.api_base_is_complex_vartype
def test_is_complex0():
    """
    int
    """
    x = paddle.to_tensor([[1, 3], [3, 2], [5, 6]])
    assert not paddle.is_complex(x)
    assert not paddle.Tensor.is_complex(x)


@pytest.mark.api_base_is_complex_parameters
def test_is_complex1():
    """
    float
    """
    x = paddle.rand((4,))
    assert not paddle.is_complex(x)
    assert not paddle.Tensor.is_complex(x)


@pytest.mark.api_base_is_complex_parameters
def test_is_complex2():
    """
    complex
    """
    types = ["complex64", "complex128"]
    z = paddle.rand((3,)) + 1j * paddle.rand((3,))
    for dtype in types:
        z = z.astype(dtype)
        assert paddle.is_complex(z)
        assert paddle.Tensor.is_complex(z)
