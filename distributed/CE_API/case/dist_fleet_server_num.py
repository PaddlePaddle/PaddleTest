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
  * @file dist_fleet_server_num.py
  * @author liyang109@baidu.com
  * @date 2021-01-20 13:07
  * @brief
  *
  **************************************************************************/
"""
import sys
import paddle.distributed.fleet as fleet
from utils import run_priority


fleet.init()


@run_priority(level="P0")
def test_server_num():
    """test_server_num"""
    assert fleet.server_num() == 2
    print("{} ... ok".format(sys._getframe().f_code.co_name))


if __name__ == "__main__":
    test_server_num()
