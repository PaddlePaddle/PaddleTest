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
  * @file dist_collective_alltoall.py
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


@run_priority(level="P0")
def test_collective_alltoall():
    """test_collective_alltoall"""
    types = [np.float16, np.float32, np.float64, np.int32, np.int64, np.int8, np.uint8]
    for t in types:
        out_tensor_list = []
        if paddle.distributed.ParallelEnv().local_rank == 0:
            np_data1 = np.array([[1, 2, 3], [4, 5, 6]]).astype(t)
            np_data2 = np.array([[7, 8, 9], [10, 11, 12]]).astype(t)
        else:
            np_data1 = np.array([[13, 14, 15], [16, 17, 18]]).astype(t)
            np_data2 = np.array([[19, 20, 21], [22, 23, 24]]).astype(t)
        data1 = paddle.to_tensor(np_data1)
        data2 = paddle.to_tensor(np_data2)
        paddle.distributed.alltoall([data1, data2], out_tensor_list)
        assert len(out_tensor_list) == 2
        print("test_collective_alltoall %s... ok" % t)


if __name__ == "__main__":
    test_collective_alltoall()
