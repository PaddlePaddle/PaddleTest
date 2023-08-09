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
  * @file dist_fleet_role.py
  * @author liujie44@baidu.com
  * @date 2021-11-15 16:07
  * @brief
  *
  **************************************************************************/
"""
import os
import sys
import paddle.distributed.fleet as fleet
from utils import run_priority


@run_priority(level="P0")
def test_dist_fleet_init_role():
    """test_dist_fleet_init_role"""
    role = fleet.PaddleCloudRoleMaker()
    fleet.init(role_maker=role)
    print("{} ... ok".format(sys._getframe().f_code.co_name))


if __name__ == "__main__":
    test_dist_fleet_init_role()
