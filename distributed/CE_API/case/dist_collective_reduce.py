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
  * @file dist_collective_reduce.py
  * @author liujie44@baidu.com
  * @date 2021-11-09 11:10
  * @brief
  *
  **************************************************************************/
"""
import sys

import numpy as np
import paddle
from paddle.distributed import ReduceOp, init_parallel_env

from utils import run_priority

types = [np.float16, np.float32, np.float64, np.int32, np.int64, np.int8, np.uint8]
paddle.set_device("gpu:%d" % paddle.distributed.ParallelEnv().dev_id)
init_parallel_env()


@run_priority(level="P0")
def test_collective_reduce_sum():
    """test_collective_reduce_sum"""
    for t in types:
        if paddle.distributed.ParallelEnv().local_rank == 0:
            np_data = np.array([[4, 5, 6], [4, 5, 6]]).astype(t)
        else:
            np_data = np.array([[1, 2, 3], [1, 2, 3]]).astype(t)
        data = paddle.to_tensor(np_data)
        paddle.distributed.reduce(data, 0, ReduceOp.SUM)
        out = data.numpy()
        assert len(out) == 2
        print("test_collective_reduce_sum %s ... ok" % t)


@run_priority(level="P0")
def test_collective_reduce_max():
    """test_collective_reduce_max"""
    for t in types:
        if paddle.distributed.ParallelEnv().local_rank == 0:
            np_data = np.array([[4, 5, 6], [4, 5, 6]]).astype(t)
        else:
            np_data = np.array([[1, 2, 3], [1, 2, 3]]).astype(t)
        data = paddle.to_tensor(np_data)
        paddle.distributed.reduce(data, 0, ReduceOp.MAX)
        out = data.numpy()
        assert len(out) == 2
        print("test_collective_reduce_max %s ... ok" % t)


@run_priority(level="P0")
def test_collective_reduce_min():
    """test_collective_reduce_min"""
    for t in types:
        if paddle.distributed.ParallelEnv().local_rank == 0:
            np_data = np.array([[4, 5, 6], [4, 5, 6]]).astype(t)
        else:
            np_data = np.array([[1, 2, 3], [1, 2, 3]]).astype(t)
        data = paddle.to_tensor(np_data)
        paddle.distributed.reduce(data, 0, ReduceOp.MIN)
        out = data.numpy()
        assert out[0][0] == 1
        assert len(out) == 2
        print("test_collective_reduce_min %s ... ok" % t)


@run_priority(level="P0")
def test_collective_reduce_prod():
    """test_collective_reduce_prod"""
    for t in types:
        if paddle.distributed.ParallelEnv().local_rank == 0:
            np_data = np.array([[4, 5, 6], [4, 5, 6]]).astype(t)
        else:
            np_data = np.array([[1, 2, 3], [1, 2, 3]]).astype(t)
        data = paddle.to_tensor(np_data)
        paddle.distributed.reduce(data, 0, ReduceOp.PROD)
        out = data.numpy()
        assert len(out) == 2
        print("test_collective_reduce_prod %s ... ok" % t)


if __name__ == "__main__":
    test_collective_reduce_max()
    test_collective_reduce_min()
    test_collective_reduce_prod()
    test_collective_reduce_sum()
