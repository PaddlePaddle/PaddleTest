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
  * @file dist_send_recv_object_list.py
  * @author lvkunpeng@baidu.com
  * @date 2025-07-15
  * @brief
  *
  **************************************************************************/
"""
import paddle
import paddle.distributed as dist
from paddle.distributed import send_object_list,recv_object_list
from utils import run_priority


@run_priority(level="P0")
def test_send_recv_object_list():
    """test_send_recv_object_list"""
    dist.init_parallel_env()
    rank = dist.get_rank()

    if rank == 0:
        data_to_send = ["hello", {"key": 100}, [1, 2, 3]]
        dist.send_object_list(data_to_send, dst=1)
    elif rank == 1:
        received_data = [None] * 3
        dist.recv_object_list(received_data, src=0)

        assert received_data == ["hello", {"key": 100}, [1, 2, 3]]


if __name__ == "__main__":
    test_send_recv_object_list()