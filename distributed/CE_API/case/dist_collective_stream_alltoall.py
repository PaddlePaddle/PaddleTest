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
  * @file dist_collective_stream_alltoall.py
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
def test_collective_stream_alltoall_default_tensor():
    """test_collective_stream_alltoall_default_tensor"""
    types = [np.float16, np.float32, np.float64, np.int32, np.int64, np.int8, np.uint8]
    for t in types:
        out_tensor_list = paddle.empty([6], dtype=t)
        if paddle.distributed.ParallelEnv().local_rank == 0:
            np_data = np.array([[1, 2, 3], [4, 5, 6]]).astype(t)
        else:
            np_data = np.array([[13, 14, 15], [16, 17, 18]]).astype(t)
        data = paddle.to_tensor(np_data)
        paddle.distributed.stream.alltoall(out_tensor_list, data)
        assert len(out_tensor_list) == 6
        print("test_collective_stream_alltoall_default_tensor %s... ok" % t)


def test_collective_stream_alltoall_sync_calc_tensor():
    """test_collective_stream_alltoall_sync_calc_tensor"""
    types = [np.float16, np.float32, np.float64, np.int32, np.int64, np.int8, np.uint8]
    for t in types:
        out_tensor_list = paddle.empty([6], dtype=t)
        if paddle.distributed.ParallelEnv().local_rank == 0:
            np_data = np.array([[1, 2, 3], [4, 5, 6]]).astype(t)
        else:
            np_data = np.array([[13, 14, 15], [16, 17, 18]]).astype(t)
        data = paddle.to_tensor(np_data)
        paddle.distributed.stream.alltoall(out_tensor_list, data, sync_op=True, use_calc_stream=True)
        assert len(out_tensor_list) == 6
        print("test_collective_stream_alltoall_sync_calc_tensor %s... ok" % t)


def test_collective_stream_alltoall_tensor():
    """test_collective_stream_alltoall_tensor"""
    types = [np.float16, np.float32, np.float64, np.int32, np.int64, np.int8, np.uint8]
    for t in types:
        out_tensor_list = paddle.empty([6], dtype=t)
        if paddle.distributed.ParallelEnv().local_rank == 0:
            np_data = np.array([[1, 2, 3], [4, 5, 6]]).astype(t)
        else:
            np_data = np.array([[13, 14, 15], [16, 17, 18]]).astype(t)
        data = paddle.to_tensor(np_data)
        paddle.distributed.stream.alltoall(out_tensor_list, data, sync_op=False, use_calc_stream=False)
        assert len(out_tensor_list) == 6
        print("test_collective_stream_alltoall_tensor %s... ok" % t)


def test_collective_stream_alltoall_stream_sync_calc_tensorlist():
    """test_collective_stream_alltoall_stream_sync_calc_tensorlist"""
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
        paddle.distributed.stream.alltoall(out_tensor_list, [data1, data2], sync_op=True, use_calc_stream=True)
        assert len(out_tensor_list) == 2
        print("test_collective_stream_alltoall_stream_sync_calc_tensorlist %s... ok" % t)


if __name__ == "__main__":
    test_collective_stream_alltoall_default_tensor()
    test_collective_stream_alltoall_sync_calc_tensor()
    test_collective_stream_alltoall_tensor()
    test_collective_stream_alltoall_stream_sync_calc_tensorlist()
