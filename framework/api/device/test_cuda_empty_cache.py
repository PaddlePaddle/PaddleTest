#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_cuda_empty_cache
"""

import paddle
import pytest


@pytest.mark.api_device_cuda_empty_cache_parameters
def test_cuda_empty_cache():
    """
    test
    """
    if not paddle.is_compiled_with_cuda():
        return
    paddle.set_device("gpu")
    a = paddle.randn([512, 512, 512], "float")
    del a
    paddle.device.cuda.empty_cache()
