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
  * @file dist_env_get_rank.py
  * @author liujie44@baidu.com
  * @date 2021-11-09 11:10
  * @brief
  *
  **************************************************************************/
"""
import sys
import paddle
import paddle.distributed as dist
from utils import run_priority


@run_priority(level="P0")
def test_env_get_rank():
    """test_env_get_rank"""
    dist.get_rank()
    print("The rank is %d" % dist.get_rank())
    print("{} ... ok".format(sys._getframe().f_code.co_name))


if __name__ == "__main__":
    test_env_get_rank()
