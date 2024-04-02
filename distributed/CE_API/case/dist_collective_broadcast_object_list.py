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
  * Copyright (c) 2024 Baidu.com, Inc. All Rights Reserved
  * @file dist_collective_broadcast_object_list.py
  * @author liujie44@baidu.com
  * @date 2024-02-21
  * @brief
  *
  **************************************************************************/
"""
import sys

import numpy as np
import paddle
import paddle.distributed as dist

dist.init_parallel_env()

from utils import run_priority

paddle.set_device("gpu:%d" % paddle.distributed.ParallelEnv().dev_id)


@run_priority(level="P0")
def test_collective_broadcast_object_list_c1():
    """test_collective_broadcast_object_list_c1"""
    if dist.get_rank() == 0:
        object_list = [{"foo": [1, 2, 3]}]
    else:
        object_list = [{"bar": [4, 5, 6]}]
    dist.broadcast_object_list(object_list, src=1)
    assert object_list[0] == {"bar": [4, 5, 6]}
    print("test_collective_broadcast_object_list_c1 ... ok")


@run_priority(level="P0")
def test_collective_broadcast_object_list_c2():
    """test_collective_broadcast_object_list_c2"""
    if dist.get_rank() == 0:
        object_list = [{"foo": [1, 2, 3]}]
    else:
        object_list = [{"bar": [4, 5, 6]}]
    dist.broadcast_object_list(object_list, src=0)
    assert object_list[0] == {"foo": [1, 2, 3]}
    print("test_collective_broadcast_object_list_c2 ... ok")


if __name__ == "__main__":
    test_collective_broadcast_object_list_c1()
    test_collective_broadcast_object_list_c2()
