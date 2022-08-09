#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_memory_allocated
"""

import os
import sys

os.environ["FLAGS_allocator_strategy"] = "auto_growth"
print(os.environ.get("FLAGS_allocator_strategy"))
sys.path.append("../../utils/")
from interceptor import skip_not_compile_gpu

import paddle
import pytest

x = paddle.to_tensor([1.0])


@skip_not_compile_gpu
@pytest.mark.api_device_memory_allocated_parameters
def test_memory_allocated0():
    """
    device = None
    """
    mem_all0 = paddle.device.cuda.memory_allocated(device=None)
    max_mem_all0 = paddle.device.cuda.max_memory_allocated(device=None)
    mem_res0 = paddle.device.cuda.memory_reserved(device=None)
    max_mem_res0 = paddle.device.cuda.max_memory_reserved(device=None)
    assert mem_all0 == 256
    assert max_mem_all0 == 256
    assert mem_res0 == 256
    assert max_mem_res0 == 256


@skip_not_compile_gpu
@pytest.mark.api_device_memory_allocated_parameters
def test_memory_allocated1():
    """
    default
    """
    mem_all0 = paddle.device.cuda.memory_allocated()
    max_mem_all0 = paddle.device.cuda.max_memory_allocated()
    mem_res0 = paddle.device.cuda.memory_reserved()
    max_mem_res0 = paddle.device.cuda.max_memory_reserved()
    assert mem_all0 == 256
    assert max_mem_all0 == 256
    assert mem_res0 == 256
    assert max_mem_res0 == 256


@skip_not_compile_gpu
@pytest.mark.api_device_memory_allocated_parameters
def test_memory_allocated2():
    """
    device = paddle.CUDAPlace(0)
    """
    mem_all0 = paddle.device.cuda.memory_allocated(device=paddle.CUDAPlace(0))
    max_mem_all0 = paddle.device.cuda.max_memory_allocated(device=paddle.CUDAPlace(0))
    mem_res0 = paddle.device.cuda.memory_reserved(device=paddle.CUDAPlace(0))
    max_mem_res0 = paddle.device.cuda.max_memory_reserved(device=paddle.CUDAPlace(0))
    assert mem_all0 == 256
    assert max_mem_all0 == 256
    assert mem_res0 == 256
    assert max_mem_res0 == 256


@skip_not_compile_gpu
@pytest.mark.api_device_memory_allocated_parameters
def test_memory_allocated3():
    """
    device = 0
    """
    mem_all0 = paddle.device.cuda.memory_allocated(device=0)
    max_mem_all0 = paddle.device.cuda.max_memory_allocated(device=0)
    mem_res0 = paddle.device.cuda.memory_reserved(device=0)
    max_mem_res0 = paddle.device.cuda.max_memory_reserved(device=0)
    assert mem_all0 == 256
    assert max_mem_all0 == 256
    assert mem_res0 == 256
    assert max_mem_res0 == 256


@skip_not_compile_gpu
@pytest.mark.api_device_memory_allocated_parameters
def test_memory_allocated4():
    """
    device = gpu:0
    """
    mem_all0 = paddle.device.cuda.memory_allocated(device="gpu:0")
    max_mem_all0 = paddle.device.cuda.max_memory_allocated(device="gpu:0")
    mem_res0 = paddle.device.cuda.memory_reserved(device="gpu:0")
    max_mem_res0 = paddle.device.cuda.max_memory_reserved(device="gpu:0")
    assert mem_all0 == 256
    assert max_mem_all0 == 256
    assert mem_res0 == 256
    assert max_mem_res0 == 256
