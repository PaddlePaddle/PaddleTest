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
  * @file dist_auto_shard_tensor.py
  * @author liujie44@baidu.com
  * @date 2021-11-09 11:00
  * @brief
  *
  **************************************************************************/
"""
import sys
import numpy as np
import paddle
from paddle.distributed.fleet import auto

from utils import run_priority

paddle.enable_static()


@run_priority(level="P0")
def test_auto_shard_tensor():
    """test_auto_shard_tensor"""
    mesh = auto.ProcessMesh([[0, 1], [2, 3]], dim_names=["x", "y"])
    x = paddle.ones([4, 6])
    shard_spec = ["x", "y"]
    auto.shard_tensor(x, mesh, shard_spec)

    print("test_auto_shard_tensor ... ok")


if __name__ == "__main__":
    test_auto_shard_tensor()
