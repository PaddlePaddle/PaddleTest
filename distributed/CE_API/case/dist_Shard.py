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
  * @file dist_Shard.py
  * @author liujie44@baidu.com
  * @date 2024-02-20
  * @brief
  *
  **************************************************************************/
"""
import paddle
import paddle.distributed as dist
from utils import run_priority


@run_priority(level="P0")
def test_Shard():
    """test_Shard"""

    mesh = dist.ProcessMesh([[2, 4, 5], [0, 1, 3]], dim_names=["x", "y"])
    a = paddle.to_tensor([[1, 2, 3], [5, 6, 7]])
    # distributed tensor
    d_tensor = dist.shard_tensor(a, mesh, [dist.Shard(0), dist.Shard(1)])
    assert d_tensor.shape == [2, 3]
    print("test_Shard ... ok")


if __name__ == "__main__":
    test_Shard()
