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
  * @file dist_utils_global_gather.py
  * @author liujie44@baidu.com
  * @date 2021-11-03 19:22
  * @brief
  *
  **************************************************************************/
"""
import os
import sys

import numpy as np
import paddle
import paddle.distributed.utils.moe_utils as moe_utils
from paddle.distributed import init_parallel_env

from utils import run_priority


@run_priority(level="P0")
def test_utils_global_gather():
    """test_utils_global_gather"""
    init_parallel_env()
    # n_expert = 2
    # world_size = 2
    # d_model = 2
    # in_feat = d_model
    if paddle.distributed.ParallelEnv().local_rank == 0:
        local_count = np.array([2, 1, 1, 1])
        global_count = np.array([2, 1, 1, 1])
    else:
        local_count = np.array([1, 1, 2, 1])
        global_count = np.array([1, 1, 2, 1])
    local_count = paddle.to_tensor(local_count, dtype="int64")
    global_count = paddle.to_tensor(global_count, dtype="int64")

    types = [np.float16, np.float32, np.float64]
    for t in types:
        local_input_buf = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]).astype(t)
        local_input_buf = paddle.to_tensor(local_input_buf, stop_gradient=False)

        res = moe_utils.global_scatter(local_input_buf, local_count, global_count)
        # out for rank 0: [[1, 2], [3, 4], [7, 8], [1, 2], [7, 8]]
        # out for rank 1: [[5, 6], [9, 10], [3, 4], [5, 6], [9, 10]]
        out = res.numpy()
        print(out)

        res.stop_gradient = False
        c = res * res
        c.backward()
        out_grad = local_input_buf.grad.numpy()
        # out for rank 0: [[2, 4], [6, 8], [10, 12], [14, 16], [18, 20]]
        # out for rank 1: [[2, 4], [6, 8], [10, 12], [14, 16], [18, 20]]
        print(out_grad)
        print("test_utils_global_gather  %s... ok" % t)


if __name__ == "__main__":
    test_utils_global_gather()
