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
  * @file dist_fleet_paddlecloudrolemaker.py
  * @author liujie44@baidu.com
  * @date 2021-11-12 15:26
  * @brief
  *
  **************************************************************************/
"""
import os
import sys
import paddle.distributed.fleet as fleet
from utils import run_priority


@run_priority(level="P0")
def test_dist_fleet_paddlecloudrolemaker():
    """test_dist_fleet_paddlecloudrolemaker"""
    os.environ["PADDLE_PSERVER_NUMS"] = "1"
    os.environ["PADDLE_TRAINERS_NUM"] = "1"

    os.environ["POD_IP"] = "127.0.0.1"
    os.environ["PADDLE_PORT"] = "36001"
    os.environ["TRAINING_ROLE"] = "PSERVER"
    os.environ["PADDLE_PSERVERS_IP_PORT_LIST"] = "127.0.0.1:36001,127.0.0.1:36001"
    os.environ["PADDLE_TRAINER_ID"] = "0"

    role = fleet.PaddleCloudRoleMaker()
    fleet.init(role)
    print(str(role.to_string()))
    assert str(role.to_string())[0:7] == "role: 2"
    assert str(role.to_string())[44:47] == "10."
    assert str(role.to_string())[-17:-8] == "127.0.0.1"
    print("{} ... ok".format(sys._getframe().f_code.co_name))


if __name__ == "__main__":
    test_dist_fleet_paddlecloudrolemaker()
