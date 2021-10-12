#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_cuda_get_device_capability
"""

import paddle
import pytest


@pytest.mark.device_cuda_get_device_capability_parameters
def test_cuda_get_capability_capability0():
    """
    default
    """
    res = paddle.device.cuda.get_device_capability()
    assert res == (6, 1)


@pytest.mark.device_cuda_get_device_capability_parameters
def test_cuda_get_capability_capability1():
    """
    device = 0
    """
    res = paddle.device.cuda.get_device_capability()
    assert res == (6, 1)


@pytest.mark.device_cuda_get_device_capability_parameters
def test_cuda_get_capability_capability2():
    """
    device = paddle.CUDAPlace(0)
    """
    res = paddle.device.cuda.get_device_capability(paddle.CUDAPlace(0))
    assert res == (6, 1)
