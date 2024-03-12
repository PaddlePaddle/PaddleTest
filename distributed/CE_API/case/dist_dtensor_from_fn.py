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
  * @file dist_dtensor_from_fn.py
  * @author liujie44@baidu.com
  * @date 2024-02-21
  * @brief
  *
  **************************************************************************/
"""
import paddle
import paddle.distributed as dist
from utils import run_priority


@run_priority(level="P0")
def test_dtensor_from_fn():
    """test_dtensor_from_fn"""
    mesh = dist.ProcessMesh([0, 1], dim_names=["x"])
    # Call the function dtensor_from_fn with dist_attr parameter
    d_tensor = dist.dtensor_from_fn(paddle.ones, mesh, [dist.Replicate()], shape=[1])
    assert d_tensor.shape == [1]

    print("test_dtensor_from_fn ... ok")


if __name__ == "__main__":
    test_dtensor_from_fn()
