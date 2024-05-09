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
  * @file dist_scatter_object_list.py
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
def test_scatter_object_list():
    """test_scatter_object_list"""

    out_object_list = []
    if dist.get_rank() == 0:
        in_object_list = [{"foo": [1, 2, 3]}, {"foo": [4, 5, 6]}]
    else:
        in_object_list = [{"bar": [1, 2, 3]}, {"bar": [4, 5, 6]}]
    dist.scatter_object_list(out_object_list, in_object_list, src=1)

    if dist.get_rank() == 0:
        assert out_object_list == [{"bar": [1, 2, 3]}]
    else:
        assert out_object_list == [{"bar": [4, 5, 6]}]
    print("test_scatter_object_list ... ok")


if __name__ == "__main__":
    test_scatter_object_list()
