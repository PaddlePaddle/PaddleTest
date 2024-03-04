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
  * @file dist_load_state_dict.py
  * @author liujie44@baidu.com
  * @date 2024-02-21
  * @brief
  *
  **************************************************************************/
"""
import os
import paddle
import paddle.distributed as dist

from utils import run_priority

dist.init_parallel_env()
os.system("rm -rf ./checkpoint")


@run_priority(level="P0")
def test_load_state_dict():
    """test_load_state_dict"""
    ckpt_path = "./checkpoint"
    w1 = paddle.arange(32).reshape([4, 8])
    mesh = dist.ProcessMesh([0, 1])
    sharded_w1 = dist.shard_tensor(w1, mesh, [dist.Shard(0)])
    state_dict = {"w1": sharded_w1}
    dist.save_state_dict(state_dict, ckpt_path)
    # w1_to_load = paddle.zeros_like(w1)
    sharded_w1_to_load = dist.shard_tensor(w1, mesh, [dist.Replicate()])
    state_dict_to_load = {"w1": sharded_w1_to_load}
    dist.load_state_dict(state_dict_to_load, ckpt_path)
    print(f"state_dict_to_load:{state_dict_to_load}")
    assert os.path.exists(ckpt_path) is True
    assert len(state_dict_to_load) == 1
    print("test_load_state_dict ... ok")


if __name__ == "__main__":
    test_load_state_dict()
