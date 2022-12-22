#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test Identity
"""
from apibase import randtool
from apibase import compare

import paddle
import pytest
import numpy as np


@pytest.mark.api_base_Identity_vartype
def test_Identity_base():
    """
    base
    """
    dtype = [np.bool8, np.int8, np.int16, np.int32, np.int64, np.float32, np.float16, np.float64]
    for i in dtype:
        x = randtool("float", -5, 5, [6, 6]).astype(i)
        res = x
        identity = paddle.nn.Identity()
        paddle_res = identity(x)
        compare(paddle_res, res)


@pytest.mark.api_base_Identity_parameters
def test_Identity_base1():
    """
    unused params
    """
    dtype = [np.bool8, np.int8, np.int16, np.int32, np.int64, np.float32, np.float16, np.float64]
    for i in dtype:
        x = randtool("float", -5, 5, [6, 6]).astype(i)
        res = x
        identity = paddle.nn.Identity(666, "88asd", sss="3333")
        paddle_res = identity(x)
        compare(paddle_res, res)
