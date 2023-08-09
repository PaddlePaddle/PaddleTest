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
  * @file dist_collective_split.py
  * @author liujie44@baidu.com
  * @date 2021-11-10 11:10
  * @brief
  *
  **************************************************************************/
"""
import numpy as np
import paddle
from utils import run_priority
from paddle.distributed import init_parallel_env
import paddle.distributed.fleet as fleet

paddle.set_device("gpu:%d" % paddle.distributed.ParallelEnv().dev_id)
init_parallel_env()
paddle.enable_static()
fleet.init()


@run_priority(level="P0")
def test_collective_split_embedding():
    """test_collective_split_embedding"""
    types = [np.int32, np.int64]
    for t in types:
        x = paddle.randint(0, 8, shape=[10, 4])
        data = paddle.randint_like(x, 0, 8, dtype=t)
        emb_out = paddle.distributed.split(data, (8, 8), operation="embedding", num_partitions=2)
        print(data)
        print(emb_out)
        print("test_collective_split_embedding %s ... ok" % t)


@run_priority(level="P0")
def test_collective_split_linear_c1():
    """test_collective_split_linear_c2"""
    types = [np.float16, np.float32, np.float64]
    for t in types:
        x = paddle.randint(0, 8, shape=[10, 4])
        data = paddle.randint_like(x, 0, 8, dtype=t)
        emb_out = paddle.distributed.split(data, (8, 8), operation="linear", num_partitions=2, axis=0)
        print(emb_out)
        print("test_collective_split_linear_c1 %s ... ok" % t)


@run_priority(level="P0")
def test_collective_split_linear_c2():
    """test_collective_split_linear_c2"""
    types = [np.float16, np.float32, np.float64]
    for t in types:
        x = paddle.randint(0, 8, shape=[10, 4])
        data = paddle.randint_like(x, 0, 8, dtype=t)
        emb_out = paddle.distributed.split(data, (8, 8), operation="linear", num_partitions=2, axis=1)
        print(emb_out)
        print("test_collective_split_linear_c2 %s ... ok" % t)


if __name__ == "__main__":
    test_collective_split_embedding()
    test_collective_split_linear_c1()
    test_collective_split_linear_c2()
