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
  * @file dist_reshard.py
  * @author liujie44@baidu.com
  * @date 2024-02-21
  * @brief
  *
  **************************************************************************/
"""
import paddle
import paddle.distributed as dist
from utils import run_priority

dist.init_parallel_env()


@run_priority(level="P0")
def test_reshard():
    """test_reshard"""
    mesh = dist.ProcessMesh([0, 1], dim_names=["x"])
    # dense tensor
    a = paddle.ones([10, 20])
    # distributed tensor
    d_tensor = dist.shard_tensor(a, mesh, [dist.Partial()])
    out_d_tensor = dist.reshard(d_tensor, mesh, [dist.Replicate()])

    assert out_d_tensor.shape == d_tensor.shape
    print("test_reshard ... ok")


if __name__ == "__main__":
    test_reshard()
