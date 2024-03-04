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
  * @file dist_gather.py
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
def test_gather():
    """test_gather"""
    gather_list = []
    if dist.get_rank() == 0:
        data = paddle.to_tensor([1, 2, 3])
        dist.gather(data, gather_list, dst=0)
        assert len(gather_list) == 2
    else:
        data1 = paddle.to_tensor([4, 5, 6])
        dist.gather(data1, gather_list, dst=0)
        assert len(gather_list) == 0

    print("test_gather ... ok")


if __name__ == "__main__":
    test_gather()
