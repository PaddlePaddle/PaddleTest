#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_cuda_get_device_name
"""

import paddle
import pytest


@pytest.mark.device_cuda_get_device_name_parameters
def test_cuda_get_device_name0():
    """
    default
    """
    if paddle.device.is_compiled_with_cuda():
        device = paddle.device.cuda.get_device_name()
        assert isinstance(device, str)


@pytest.mark.device_cuda_get_device_name_parameters
def test_cuda_get_device_name1():
    """
    device = 0
    """
    if paddle.device.is_compiled_with_cuda():
        device = paddle.device.cuda.get_device_name(0)
        assert isinstance(device, str)


@pytest.mark.device_cuda_get_device_name_parameters
def test_cuda_get_device_name2():
    """
    device = paddle.CUDAPlace(0)
    """
    if paddle.device.is_compiled_with_cuda():
        device = paddle.device.cuda.get_device_name(paddle.CUDAPlace(0))
        assert isinstance(device, str)
