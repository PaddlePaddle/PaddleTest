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
  * @file dist_collective_all_gather_object.py
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
def test_collective_all_gather_object():
    """test_collective_all_gather"""
    types = [np.float16, np.float32, np.float64, np.int32, np.int64, np.int8, np.uint8, np.complex64, np.complex128]
    for t in types:
        tensor_list = []
        if paddle.distributed.ParallelEnv().local_rank == 0:
            obj = {"foo": [1, 2, 3]}
            paddle.distributed.all_gather_object(tensor_list, obj)
        else:
            obj = {"bar": [4, 5, 6]}
            paddle.distributed.all_gather_object(tensor_list, obj)
        assert len(tensor_list) == 2
        print("test_collective_all_gather_object %s... ok" % t)


if __name__ == "__main__":
    test_collective_all_gather_object()
