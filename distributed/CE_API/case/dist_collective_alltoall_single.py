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
  * @file dist_collective_alltoall_single.py
  * @author liujie44@baidu.com
  * @date 2021-11-09 11:00
  * @brief
  *
  **************************************************************************/
"""
import sys

import numpy as np
import paddle
from paddle.distributed import init_parallel_env

from utils import run_priority

paddle.set_device("gpu:%d" % paddle.distributed.ParallelEnv().dev_id)
init_parallel_env()
rank = paddle.distributed.get_rank()
size = paddle.distributed.get_world_size()


@run_priority(level="P0")
def test_collective_alltoall_single1():
    """test_collective_alltoall_single1"""
    types = [np.float16, np.float32, np.float64, np.int32, np.int64, np.int8, np.uint8]
    for t in types:
        # data = paddle.arange(2, dtype=t) + rank * 2
        np_data = np.array([0, 1]).astype(t)
        data = paddle.to_tensor(np_data)
        output = paddle.empty([2], dtype=t)
        paddle.distributed.alltoall_single(data, output)
        print(output)
        assert len(output) == 2
        print("test_collective_alltoall_single1 %s... ok" % t)


@run_priority(level="P0")
def test_collective_alltoall_single2():
    """test_collective_alltoall_single2"""
    types = [np.float16, np.float32, np.float64, np.int32, np.int64, np.int8, np.uint8]
    for t in types:
        in_split_sizes = [i + 1 for i in range(size)]
        # in_split_sizes for rank 0: [1, 2]
        # in_split_sizes for rank 1: [1, 2]
        out_split_sizes = [rank + 1 for i in range(size)]
        # out_split_sizes for rank 0: [1, 1]
        # out_split_sizes for rank 1: [2, 2]
        # data = paddle.ones([sum(in_split_sizes), size], dtype=t) * rank
        np_data = np.array([[0, 0], [0, 0], [0, 0]]).astype(t)
        data = paddle.to_tensor(np_data)
        output = paddle.empty([(rank + 1) * size, size], dtype=t)
        group = paddle.distributed.new_group([0, 1])
        task = paddle.distributed.alltoall_single(
            data, output, in_split_sizes, out_split_sizes, sync_op=False, group=group
        )
        task.wait()
        print(output)
        print("test_collective_alltoall_single2 %s... ok" % t)


if __name__ == "__main__":
    test_collective_alltoall_single1()
    test_collective_alltoall_single2()
