#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
# ======================================================================
#
# Copyright (c) 2017 Baidu.com, Inc. All Rights Reserved
#
# ======================================================================
"""
/***************************************************************************
  *
  * Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
  * @file utils.py
  * @author liyang109@baidu.com
  * @date 2021-01-18 18:31
  * @brief
  *
  **************************************************************************/
"""
#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
# ======================================================================
#
# Copyright (c) 2017 Baidu.com, Inc. All Rights Reserved
#
# ======================================================================
import os
import sys
import decorator
import nose.tools as tools


def run_priority(level):
    """testcase priority, with P0、P1."""

    @decorator.decorator
    def wrapper(func, *args, **kwargs):
        """wrapper"""
        if os.getenv("RUN_PRIORITY", "P0") == level:
            print("{} ... begin".format(func.__name__))
            return func(*args, **kwargs)
        else:
            return

    return wrapper


def check_data(real, expect=None, delta=None):
    """
    校验结果数据.
    Args:
        loss (list): the loss will be checked.
        delta (float):
        expect (list):
    """
    if expect:
        expect_data = expect
    else:
        expect_data = real
    length = len(expect_data)
    if delta:
        for i in range(length):
            tools.assert_almost_equal(real[i], expect_data[i], delta=delta)
    else:
        for i in range(length):
            tools.assert_equal(real[i], expect_data[i])
