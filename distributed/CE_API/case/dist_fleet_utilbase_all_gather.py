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
  * @file dist_fleet_utilbase_all_gather.py
  * @author liujie44@baidu.com
  * @date 2021-11-22 19:06
  * @brief
  *
  **************************************************************************/
"""
import sys
import os

import paddle.distributed.fleet as fleet
from paddle.distributed.fleet import PaddleCloudRoleMaker
from utils import run_priority

os.environ["PADDLE_WITH_GLOO"] = "2"


@run_priority(level="P0")
def test_all_gatcher():
    """test all gather"""
    role = PaddleCloudRoleMaker(is_collective=False, init_gloo=True, path="./tmp_gloo")
    fleet.init(role)

    if fleet.is_server():
        input = fleet.server_index()
        output = fleet.util.all_gather(input, "server")
        assert output == 0
    elif fleet.is_worker():
        input = fleet.worker_index()
        output = fleet.util.all_gather(input, "worker")
        assert output == 0
    print(output)
    output = fleet.util.all_gather(input, "all")
    print(output)
    assert output == 0


if __name__ == "__main__":
    test_all_gatcher()
