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
  * Copyright (c) 2024 Baidu.com, Inc. All Rights Reserved
  * @file dist_ParallelMode.py
  * @author liujie44@baidu.com
  * @date 2024-02-20
  * @brief
  *
  **************************************************************************/
"""
import paddle
from utils import run_priority


@run_priority(level="P0")
def test_ParallelMode():
    """test_ParallelMode"""
    parallel_mode = paddle.distributed.ParallelMode
    assert parallel_mode.DATA_PARALLEL == 0
    print("test_ParallelMode ... ok")


if __name__ == "__main__":
    test_ParallelMode()
