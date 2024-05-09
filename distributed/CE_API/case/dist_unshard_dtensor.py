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
  * @file dist_unshard_dtensor.py
  * @author liujie44@baidu.com
  * @date 2024-02-21
  * @brief
  *
  **************************************************************************/
"""
import paddle
import paddle.distributed as dist
from paddle.distributed import Replicate, Shard
from utils import run_priority


@run_priority(level="P0")
def test_unshard_dtensor():
    """test_unshard_dtensor"""

    mesh = dist.ProcessMesh([0, 1], dim_names=["x"])
    original_tensor = paddle.rand([4, 1024, 512])
    dist_tensor = dist.shard_tensor(original_tensor, mesh, [Shard(0)])
    # dense_tensor's shape is the same as original_tensor
    dense_tensor = dist.unshard_dtensor(dist_tensor)
    assert dense_tensor.shape == original_tensor.shape

    print("test_unshard_dtensor ... ok")


if __name__ == "__main__":
    test_unshard_dtensor()
