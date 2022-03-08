#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_clone
"""

import paddle
import pytest
import numpy as np


@pytest.mark.api_base_clone_parameters
def test_clone0():
    """
    x: 1d-tensor
    """
    xp = np.ones((3,))
    x = paddle.to_tensor(xp)
    x.stop_gradient = False
    clone_x = paddle.clone(x)
    y = clone_x ** 3
    y.backward()
    assert np.allclose(x.numpy(), clone_x.numpy())
    assert np.allclose(x.grad.numpy(), clone_x.grad.numpy())


@pytest.mark.api_base_clone_parameters
def test_clone1():
    """
    x: 2d-tensor
    """
    xp = np.ones((3, 3))
    x = paddle.to_tensor(xp)
    x.stop_gradient = False
    clone_x = paddle.clone(x)
    y = clone_x ** 3
    y.backward()
    assert np.allclose(x.numpy(), clone_x.numpy())
    assert np.allclose(x.grad.numpy(), clone_x.grad.numpy())


@pytest.mark.api_base_clone_parameters
def test_clone2():
    """
    x: 3d-tensor
    """
    xp = np.ones((3, 3, 4))
    x = paddle.to_tensor(xp)
    x.stop_gradient = False
    clone_x = paddle.clone(x)
    y = clone_x ** 3
    y.backward()
    assert np.allclose(x.numpy(), clone_x.numpy())
    assert np.allclose(x.grad.numpy(), clone_x.grad.numpy())


@pytest.mark.api_base_clone_parameters
def test_clone3():
    """
    x: 4d-tensor
    """
    xp = np.ones((3, 3, 4, 4))
    x = paddle.to_tensor(xp)
    x.stop_gradient = False
    clone_x = paddle.clone(x)
    y = clone_x ** 3
    y.backward()
    assert np.allclose(x.numpy(), clone_x.numpy())
    assert np.allclose(x.grad.numpy(), clone_x.grad.numpy())
