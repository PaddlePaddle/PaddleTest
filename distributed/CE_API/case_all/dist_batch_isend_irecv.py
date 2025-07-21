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
  * @file dist_batch_isend_irecv.py
  * @author lvkunpeng@baidu.com
  * @date 2025-07-14
  * @brief
  *
  **************************************************************************/
"""
import paddle
import paddle.distributed as dist
from paddle.distributed import batch_isend_irecv
from utils import run_priority

@run_priority(level="P0")
def test_batch_isend_irecv():
    """test_batch_isend_irecv"""
    dist.init_parallel_env()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    send_t = paddle.arange(2) + rank
    recv_t = paddle.empty(shape=[2], dtype=send_t.dtype)

    send_op = dist.P2POp(dist.isend, send_t, (rank + 1) % world_size)
    recv_op = dist.P2POp(dist.irecv, recv_t, (rank - 1 + world_size) % world_size)

    tasks = batch_isend_irecv([send_op, recv_op])

    for task in tasks:
        task.wait()

    expected = paddle.arange(2) + ((rank - 1 + world_size) % world_size)
    assert paddle.allclose(recv_t, expected)
    print(f"Rank {rank} passed with recv: {recv_t.numpy()}")

    print("test_batch_isend_irecv ... ok") 


if __name__ == '__main__':
    test_batch_isend_irecv()