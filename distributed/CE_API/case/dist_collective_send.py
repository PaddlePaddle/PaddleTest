#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
/***************************************************************************
  *
  * Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
  * @file dist_collective_send.py
  * @author liujie44@baidu.com
  * @date 2021-11-09 11:00
  * @brief
  *
  **************************************************************************/
"""
import numpy as np
import paddle
from paddle.distributed import init_parallel_env

from utils import run_priority

types = [np.float16, np.float32, np.float64, np.int32, np.int64, np.int8, np.uint8]
init_parallel_env()


@run_priority(level="P0")
def test_collective_send1():
    """test_collective_send1"""
    for t in types:
        if paddle.distributed.ParallelEnv().rank == 0:
            np_data = np.array([7, 8, 9]).astype(t)
            data = paddle.to_tensor(np_data)
            paddle.distributed.send(data, dst=1)
        else:
            np_data = np.array([1, 2, 3]).astype(t)
            data = paddle.to_tensor(np_data)
            paddle.distributed.recv(data, src=0)
        out = data.numpy()
        assert out[0] == 7
        assert out[1] == 8
        assert out[2] == 9
        print("test_collective_send1 %s... ok" % t)


@run_priority(level="P0")
def test_collective_send2():
    """test_collective_send2"""
    for t in types:
        if paddle.distributed.ParallelEnv().rank == 0:
            np_data = np.array([7, 8, 9]).astype(t)
            data = paddle.to_tensor(np_data)
            paddle.distributed.send(data, dst=1, group=None, sync_op=True)
        else:
            np_data = np.array([1, 2, 3]).astype(t)
            data = paddle.to_tensor(np_data)
            paddle.distributed.recv(data, src=0)
        out = data.numpy()
        assert out[0] == 7
        assert out[1] == 8
        assert out[2] == 9
        print("test_collective_send2 %s... ok" % t)


@run_priority(level="P0")
def test_collective_send3():
    """test_collective_send3"""
    for t in types:
        if paddle.distributed.ParallelEnv().rank == 0:
            np_data = np.array([7, 8, 9]).astype(t)
            data = paddle.to_tensor(np_data)
            paddle.distributed.send(data, dst=1, group=None, sync_op=False)
        else:
            np_data = np.array([1, 2, 3]).astype(t)
            data = paddle.to_tensor(np_data)
            paddle.distributed.recv(data, src=0)
        out = data.numpy()
        assert out[0] == 7
        assert out[1] == 8
        assert out[2] == 9
        print("test_collective_send3 %s... ok" % t)


if __name__ == "__main__":
    test_collective_send1()
    test_collective_send2()
    test_collective_send3()
