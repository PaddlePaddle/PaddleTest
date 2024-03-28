#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
# ======================================================================
#
# Copyright (c) 2024 Baidu.com, Inc. All Rights Reserved
#
# ======================================================================
"""
/***************************************************************************
  *
  * Copyright (c) 2024 Baidu.com, Inc. All Rights Reserved
  * @file dist_save_state_dict.py
  * @author liujie44@baidu.com
  * @date 2024-02-21
  * @brief
  *
  **************************************************************************/
"""
import os

os.system("rm -rf ./checkpoint")

import paddle
import paddle.distributed as dist
from utils import run_priority

dist.init_parallel_env()


@run_priority(level="P0")
def test_save_state_dict():
    """test_save_state_dict"""

    w1 = paddle.arange(32).reshape([4, 8])
    mesh = dist.ProcessMesh([0, 1])
    sharded_w1 = dist.shard_tensor(w1, mesh, [dist.Shard(0), dist.Replicate()])
    state_dict = {"w1": sharded_w1}
    dist.save_state_dict(state_dict, "./checkpoint")

    assert os.path.exists("./checkpoint") is True
    print("test_save_state_dict ... ok")


if __name__ == "__main__":
    test_save_state_dict()
