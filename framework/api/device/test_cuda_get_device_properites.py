#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_cuda_get_device_properties
"""

import paddle
import pytest


@pytest.mark.api_device_cuda_get_device_propertiesparameters
def test_cuda_get_device_properties0():
    """
    test0
    """
    if not paddle.is_compiled_with_cuda():
        return
    paddle.device.cuda.get_device_properties()


@pytest.mark.api_device_cuda_get_device_propertiesparameters
def test_cuda_get_device_properties1():
    """
    test1
    """
    if not paddle.is_compiled_with_cuda():
        return
    paddle.device.cuda.get_device_properties(0)


@pytest.mark.api_device_cuda_get_device_propertiesparameters
def test_cuda_get_device_properties2():
    """
    test2
    """
    if not paddle.is_compiled_with_cuda():
        return
    paddle.device.cuda.get_device_properties("gpu: 0")


@pytest.mark.api_device_cuda_get_device_propertiesparameters
def test_cuda_get_device_properties3():
    """
    test3
    """
    if not paddle.is_compiled_with_cuda():
        return
    paddle.device.cuda.get_device_properties(paddle.CUDAPlace(0))
