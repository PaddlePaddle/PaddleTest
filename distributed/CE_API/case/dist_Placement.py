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
  * @file dist_Placement.py
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
def test_Placement():
    """test_Placement"""
    placements = [dist.Replicate(), dist.Shard(0), dist.Partial()]
    for p in placements:
        if isinstance(p, dist.Placement):
            if p.is_replicated():
                print("replicate.")
            elif p.is_shard():
                print("shard.")
            elif p.is_partial():
                print("partial.")
    print("test_Placement ... ok")


if __name__ == "__main__":
    test_Placement()
