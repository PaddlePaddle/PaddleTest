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
  * @file dist_collective_stream_all_gather.py
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
def test_collective_stream_all_gather_default_tensor():
    """test_collective_stream_all_gather_default_tensor"""
    types = [np.float16, np.float32, np.float64, np.int32, np.int64, np.int8, np.uint8]
    for t in types:
        tensor = paddle.to_tensor([0] * 12, dtype=t)
        if paddle.distributed.ParallelEnv().local_rank == 0:
            np_data1 = np.array([[4, 5, 6], [4, 5, 6]]).astype(t)
            np_data2 = np.array([[4, 5, 6], [4, 5, 6]]).astype(t)
            data1 = paddle.to_tensor(np_data1)
            data2 = paddle.to_tensor(np_data2)
            paddle.distributed.stream.all_gather(tensor, data1)
        else:
            np_data1 = np.array([[1, 2, 3], [1, 2, 3]]).astype(t)
            np_data2 = np.array([[1, 2, 3], [1, 2, 3]]).astype(t)
            data1 = paddle.to_tensor(np_data1)
            data2 = paddle.to_tensor(np_data2)
            paddle.distributed.stream.all_gather(tensor, data2)
        out = tensor.numpy()
        assert out[0] == 4
        assert len(tensor) == 12
        print("test_collective_stream_all_gather_default_tensor %s... ok" % t)


@run_priority(level="P0")
def test_collective_stream_all_gather_sync_calc_tensor():
    """test_collective_stream_all_gather_sync_calc_tensor"""
    types = [np.float16, np.float32, np.float64, np.int32, np.int64, np.int8, np.uint8]
    for t in types:
        tensor = paddle.to_tensor([0] * 12, dtype=t)
        if paddle.distributed.ParallelEnv().local_rank == 0:
            np_data1 = np.array([[4, 5, 6], [4, 5, 6]]).astype(t)
            np_data2 = np.array([[4, 5, 6], [4, 5, 6]]).astype(t)
            data1 = paddle.to_tensor(np_data1)
            data2 = paddle.to_tensor(np_data2)
            paddle.distributed.stream.all_gather(tensor, data1, sync_op=True, use_calc_stream=True)
        else:
            np_data1 = np.array([[1, 2, 3], [1, 2, 3]]).astype(t)
            np_data2 = np.array([[1, 2, 3], [1, 2, 3]]).astype(t)
            data1 = paddle.to_tensor(np_data1)
            data2 = paddle.to_tensor(np_data2)
            paddle.distributed.stream.all_gather(tensor, data2, sync_op=True, use_calc_stream=True)
        out = tensor.numpy()
        assert out[0] == 4
        assert len(tensor) == 12
        print("test_collective_stream_all_gather_sync_calc_tensor %s... ok" % t)


@run_priority(level="P0")
def test_collective_stream_all_gather_sync_calc_tensorlist():
    """test_collective_stream_all_gather_sync_calc_tensorlist"""
    types = [np.float16, np.float32, np.float64, np.int32, np.int64, np.int8, np.uint8]
    for t in types:
        tensor_list = []
        if paddle.distributed.ParallelEnv().local_rank == 0:
            np_data1 = np.array([[4, 5, 6], [4, 5, 6]]).astype(t)
            np_data2 = np.array([[4, 5, 6], [4, 5, 6]]).astype(t)
            data1 = paddle.to_tensor(np_data1)
            data2 = paddle.to_tensor(np_data2)
            paddle.distributed.stream.all_gather(tensor_list, data1, sync_op=True, use_calc_stream=True)
        else:
            np_data1 = np.array([[1, 2, 3], [1, 2, 3]]).astype(t)
            np_data2 = np.array([[1, 2, 3], [1, 2, 3]]).astype(t)
            data1 = paddle.to_tensor(np_data1)
            data2 = paddle.to_tensor(np_data2)
            paddle.distributed.stream.all_gather(tensor_list, data2, sync_op=True, use_calc_stream=True)
        out1 = tensor_list[0].numpy()
        out2 = tensor_list[1].numpy()
        assert out1[0][0] == 4
        assert out2[0][0] == 1
        assert len(tensor_list) == 2
        print("test_collective_stream_all_gather_sync_calc_tensorlist %s... ok" % t)


@run_priority(level="P0")
def test_collective_stream_all_gather_tensor():
    """test_collective_stream_all_gather_tensor"""
    types = [np.float16, np.float32, np.float64, np.int32, np.int64, np.int8, np.uint8]
    for t in types:
        tensor = paddle.to_tensor([0] * 12, dtype=t)
        if paddle.distributed.ParallelEnv().local_rank == 0:
            np_data1 = np.array([[4, 5, 6], [4, 5, 6]]).astype(t)
            np_data2 = np.array([[4, 5, 6], [4, 5, 6]]).astype(t)
            data1 = paddle.to_tensor(np_data1)
            data2 = paddle.to_tensor(np_data2)
            paddle.distributed.stream.all_gather(tensor, data1, sync_op=False, use_calc_stream=False)
        else:
            np_data1 = np.array([[1, 2, 3], [1, 2, 3]]).astype(t)
            np_data2 = np.array([[1, 2, 3], [1, 2, 3]]).astype(t)
            data1 = paddle.to_tensor(np_data1)
            data2 = paddle.to_tensor(np_data2)
            paddle.distributed.stream.all_gather(tensor, data2, sync_op=False, use_calc_stream=False)
        out = tensor.numpy()
        assert out[0] == 4
        assert len(tensor) == 12
        print("test_collective_stream_all_gather_tensor %s... ok" % t)


if __name__ == "__main__":
    test_collective_stream_all_gather_default_tensor()
    test_collective_stream_all_gather_sync_calc_tensor()
    test_collective_stream_all_gather_sync_calc_tensorlist()
    test_collective_stream_all_gather_tensor()
