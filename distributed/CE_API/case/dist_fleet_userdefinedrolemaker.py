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
  * @file dist_fleet_userdefinedrolemaker.py
  * @author liujie44@baidu.com
  * @date 2021-11-12 16:02
  * @brief
  *
  **************************************************************************/
"""
import sys
import paddle.distributed.fleet as fleet
from paddle.distributed.fleet.base.role_maker import Role
from utils import run_priority


@run_priority(level="P0")
def test_dist_fleet_UserDefinedRoleMaker():
    """test_dist_fleet_UserDefinedRoleMaker"""
    role = fleet.UserDefinedRoleMaker(
        current_id=0, role=Role.SERVER, worker_num=2, server_endpoints=["127.0.0.1:36011", "127.0.0.1:36012"]
    )
    fleet.init(role)
    print(role.to_string())
    print(str(role.to_string()))
    assert str(role.to_string())[0:7] == "role: 2"
    print(str(role.to_string())[66:75])
    assert str(role.to_string())[66:75] == "127.0.0.1"
    print("{} ... ok".format(sys._getframe().f_code.co_name))


if __name__ == "__main__":
    test_dist_fleet_UserDefinedRoleMaker()
