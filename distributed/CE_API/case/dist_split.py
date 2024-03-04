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
  * @file dist_split.py
  * @author liujie44@baidu.com
  * @date 2024-02-21
  * @brief
  *
  **************************************************************************/
"""
import paddle
import paddle.distributed.fleet as fleet
from utils import run_priority

paddle.enable_static()


@run_priority(level="P0")
def test_split():
    """test_split"""
    paddle.set_device("gpu:%d" % paddle.distributed.get_rank())
    fleet.init(is_collective=True)
    data = paddle.randint(0, 8, shape=[10, 4])
    emb_out = paddle.distributed.split(data, (8, 8), operation="embedding", num_partitions=2)
    assert emb_out.shape == (10, 4, 8)

    print("test_split ... ok")


if __name__ == "__main__":
    test_split()
