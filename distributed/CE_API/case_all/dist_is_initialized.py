#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
# ======================================================================
#
# Copyright (c) 2024 Baidu.com, Inc. All Rights Reserved
#
# ======================================================================
"""
/***************************************************************************
  *
  * Copyright (c) 2025 Baidu.com, Inc. All Rights Reserved
  * @file dist_is_initialized.py
  * @author lvkunpeng@baidu.com
  * @date 2025-07-04
  * @brief
  *
  **************************************************************************/
"""

import paddle
import paddle.distributed as dist
from paddle.distributed import is_initialized
from utils import run_priority


@run_priority(level="P0")
def test_is_initialized():
    """test_is_initialized"""
    assert is_initialized() is False
    dist.init_parallel_env()
    assert is_initialized() is True

    print("test_is_initialized ... ok")


if __name__ == "__main__":
    test_is_initialized()

