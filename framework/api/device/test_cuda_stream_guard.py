#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_cuda_stream_guard
"""

import paddle
import pytest


@pytest.mark.api_device_cuda_stream_guard_parameters
def test_cuda_stream_guard():
    """
    test
    """
    if not paddle.is_compiled_with_cuda():
        return
    s = paddle.device.cuda.Stream()
    data1 = paddle.ones(shape=[20])
    data2 = paddle.ones(shape=[20])
    with paddle.device.cuda.stream_guard(s):
        data3 = data1 + data2
    return data3
