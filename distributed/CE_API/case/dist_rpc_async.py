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
  * @file dist_rpc_async.py
  * @author liujie44@baidu.com
  * @date 2023-08-22 11:00
  * @brief
  *
  **************************************************************************/
"""
import sys
import numpy as np

import paddle
import paddle.distributed.rpc as rpc

from utils import run_priority


def add(a, b):
    """fn"""
    return a + b


def sub(a, b):
    """fn"""
    return a - b


@run_priority(level="P0")
def test_rpc_async():
    """test_rpc_async"""
    if paddle.distributed.ParallelEnv().local_rank == 0:
        rpc.init_rpc("worker0", rank=0, world_size=2, master_endpoint="127.0.0.1:8003")
        fut = rpc.rpc_async("worker0", add, args=(2, 3))
        assert fut.wait() == 5
    else:
        rpc.init_rpc("worker1", rank=1, world_size=2, master_endpoint="127.0.0.1:8003")
        fut = rpc.rpc_async("worker1", sub, args=(3, 2))
        assert fut.wait() == 1
    rpc.shutdown()

    print("test_rpc_async ... ok")


if __name__ == "__main__":
    test_rpc_async()
