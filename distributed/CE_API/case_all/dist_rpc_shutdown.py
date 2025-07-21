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
  * @file dist_rpc_shutdown.py
  * @author lvkunpeng@baidu.com
  * @date 2025-07-04
  * @brief
  *
  **************************************************************************/
"""
import paddle
import paddle.distributed as dist
import paddle.distributed.rpc as rpc

from utils import run_priority


def dummy_func(x):
    return x + 1

@run_priority(level="P0")
def test_rpc_shutdown():
    """test_rpc_shutdown"""
    local_rank = paddle.distributed.ParallelEnv().local_rank
    if local_rank == 0:
        rpc.init_rpc("worker0", rank=0, world_size=2, master_endpoint="127.0.0.1:8002")
    else:
        rpc.init_rpc("worker1", rank=1, world_size=2, master_endpoint="127.0.0.1:8002")

    rpc.shutdown()

    try:
        rpc.rpc_sync("worker0", dummy_func, args=(42,))
        assert False
    except Exception as e:
        print(f"[Rank {local_rank}] Expected exception after shutdown: {e}")
        assert "rpcagent" in str(e).lower() or "init rpc" in str(e).lower()


    print("test_rpc_shutdown ... ok")


if __name__ == '__main__':
    test_rpc_shutdown()