#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
device_count
"""

import paddle
import pytest


@pytest.mark.api_device_device_count_parameters
def test_device_count():
    """
    base
    """
    res = paddle.device.cuda.device_count()
    assert isinstance(res, int)


@pytest.mark.api_device_device_count_parameters
def test_device_count1():
    """
    Stable test
    """
    res1 = paddle.device.cuda.device_count()
    res2 = paddle.device.cuda.device_count()
    assert isinstance(res1, int)
    assert isinstance(res2, int)
    assert res1 == res2
