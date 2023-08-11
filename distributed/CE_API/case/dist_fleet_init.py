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
  * @file dist_fleet_init.py
  * @author liujie44@baidu.com
  * @date 2021-11-15 16:00
  * @brief
  *
  **************************************************************************/
"""
import os
import sys
import paddle.distributed.fleet as fleet
from utils import run_priority


@run_priority(level="P0")
def test_dist_fleet_init():
    """test_dist_fleet_init"""
    fleet.init()
    print("{} ... ok".format(sys._getframe().f_code.co_name))


if __name__ == "__main__":
    test_dist_fleet_init()
