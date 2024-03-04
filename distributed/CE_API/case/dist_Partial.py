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
  * @file dist_Partial.py
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
def test_Partial():
    """test_Partial"""
    mesh = dist.ProcessMesh([0, 1], dim_names=["x"])
    a = paddle.ones([10, 20])
    # distributed tensor
    d_tensor = dist.shard_tensor(a, mesh, [dist.Partial()])
    assert d_tensor.shape == [10, 20]
    print("test_Partial ... ok")


if __name__ == "__main__":
    test_Partial()
