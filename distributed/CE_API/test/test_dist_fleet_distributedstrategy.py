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
  * @file test_dist_fleet_distributedstrategy.py
  * @author liujie44@baidu.com
  * @date 2021-11-12 14:41
  * @brief
  *
  **************************************************************************/
"""
import os
import subprocess
from utils import run_priority

os.system("export CUDA_VISIBLE_DEVICES=0,1")


class TestApi(object):
    """test_dist_fleet_DistributedStrategy"""

    def test_dist_fleet_DistributedStrategy(self):
        """test_dist_fleet_DistributedStrategy"""
        cmd = "python -m paddle.distributed.launch --devices 0,1 --job_id distributedstrategy \
        dist_fleet_distributedstrategy.py"
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        print(out)
        pro.wait()
        pro.returncode == 0
        assert str(out).find("Error") == -1
        assert str(err).find("Error") == -1

    def test_dist_fleet_distributed_model(self):
        """test_dist_fleet_DistributedStrategy"""
        cmd = "python -m paddle.distributed.launch --devices 0,1 --job_id distributed_model \
          dist_fleet_distributed_model.py"
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        print(out)
        pro.wait()
        pro.returncode == 0
        assert str(out).find("Error") == -1
        assert str(err).find("Error") == -1
