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
  * @file dist_fleet_utilbase_all_reduce.py
  * @author liujie44@baidu.com
  * @date 2021-11-22 19:26
  * @brief
  *
  **************************************************************************/
"""
import os
import numpy as np

import paddle.distributed.fleet as fleet
from paddle.distributed.fleet import PaddleCloudRoleMaker
from utils import run_priority

os.environ["PADDLE_WITH_GLOO"] = "2"


@run_priority(level="P0")
def test_all_reduce():
    """test fleet utilBase all reduce"""
    role = PaddleCloudRoleMaker(is_collective=False, init_gloo=True, path="./tmp_gloo")
    fleet.init(role)

    if fleet.is_server():
        input = [1, 2]
        output = fleet.util.all_reduce(input, "sum", "server")
        print(output[0])
        assert output[0] == 1
    elif fleet.is_worker():
        input = np.array([3, 4])
        output = fleet.util.all_reduce(input, "sum", "worker")
        print(output[0])
        assert output[0] == 3
    output = fleet.util.all_reduce(input, "sum", "all")
    print(output)
    # assert output[0] == 4


if __name__ == "__main__":
    test_all_reduce()
