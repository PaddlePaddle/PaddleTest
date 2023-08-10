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
  * @file dist_collective_barrier.py
  * @author liujie44@baidu.com
  * @date 2021-11-09 11:10
  * @brief
  *
  **************************************************************************/
"""
import sys

import paddle
from paddle.distributed import init_parallel_env

from utils import run_priority


@run_priority(level="P0")
def test_collective_barrier():
    """test_collective_barrier"""
    paddle.set_device("gpu:%d" % paddle.distributed.ParallelEnv().dev_id)
    init_parallel_env()
    paddle.distributed.barrier()
    print("test_collective_barrier ... ok")


if __name__ == "__main__":
    test_collective_barrier()
