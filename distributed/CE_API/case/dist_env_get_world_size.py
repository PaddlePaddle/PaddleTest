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
  * @file dist_env_get_world_size.py
  * @author liujie44@baidu.com
  * @date 2021-11-09 11:10
  * @brief
  *
  **************************************************************************/
"""
import sys
import paddle.distributed as dist
from utils import run_priority


@run_priority(level="P0")
def test_env_get_world_size():
    """test_env_get_world_size"""
    assert dist.get_world_size() == 2
    print("{} ... ok".format(sys._getframe().f_code.co_name))


if __name__ == "__main__":
    test_env_get_world_size()
