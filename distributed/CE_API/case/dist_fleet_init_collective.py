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
  * @author liyang109@baidu.com
  * @date 2021-01-18 16:07
  * @brief
  *
  **************************************************************************/
"""
import os
import sys
import paddle.distributed.fleet as fleet
from utils import run_priority


@run_priority(level="P0")
def test_dist_fleet_init_collective():
    """test_dist_fleet_init_collective"""
    fleet.init(is_collective=True)
    print("{} ... ok".format(sys._getframe().f_code.co_name))


if __name__ == "__main__":
    test_dist_fleet_init_collective()
