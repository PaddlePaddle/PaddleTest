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
  * @file dist_rpc_get_current_worker_info.py
  * @author liujie44@baidu.com
  * @date 2023-08-22 11:00
  * @brief
  *
  **************************************************************************/
"""
import sys
import os
import numpy as np

import paddle
import paddle.distributed.rpc as rpc

from utils import run_priority


@run_priority(level="P0")
def test_rpc_get_current_worker_info():
    """test_rpc_get_current_worker_info"""

    if paddle.distributed.ParallelEnv().local_rank == 0:
        os.environ["PADDLE_WORKER_ENDPOINT"] = "127.0.0.1:9002"
        rpc.init_rpc("worker0", rank=0, world_size=2, master_endpoint="127.0.0.1:8004")
        print(rpc.get_current_worker_info())
    else:
        os.environ["PADDLE_WORKER_ENDPOINT"] = "127.0.0.1:9003"
        rpc.init_rpc("worker1", rank=1, world_size=2, master_endpoint="127.0.0.1:8004")
        print(rpc.get_current_worker_info())

    rpc.shutdown()

    print("test_rpc_get_current_worker_info ... ok")


if __name__ == "__main__":
    test_rpc_get_current_worker_info()
