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
  * @file dist_DistAttr.py
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
def test_DistAttr():
    """test_DistAttr"""
    mesh = dist.ProcessMesh([[2, 4, 5], [0, 1, 3]], dim_names=["x", "y"])
    dist_attr = dist.DistAttr(mesh=mesh, sharding_specs=["x", "y"])

    print(dist_attr)
    print("test_DistAttr ... ok")


if __name__ == "__main__":
    test_DistAttr()
