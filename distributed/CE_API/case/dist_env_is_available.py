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
  * @file dist_env_is_available.py
  * @author liujie44@baidu.com
  * @date 2023-08-03 14:56
  * @brief
  *
  **************************************************************************/
"""
import sys
import numpy as np
import paddle


def test_is_available():
    """test_is_available"""
    paddle.distributed.is_available()
    assert paddle.distributed.is_available() is True
    print("test_is_available... ok")


if __name__ == "__main__":
    test_is_available()
