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
  * @file test_dist_fleet_utils.py
  * @author liujie44@baidu.com
  * @date 2021-11-15 11:15
  * @brief
  *
  **************************************************************************/
"""
import os
import sys
import subprocess


class TestDistFleetUtilsApi(object):
    """TestDistFleetUtilsApi"""

    # def test_dist_fleet_utils_hdfs_client(self):
    #     """test_dist_fleet_utils_hdfs_client"""
    #     cmd = 'python -m paddle.distributed.launch --devices 0,1 --job_id hdfs_client dist_fleet_utils_hdfs_client.py'
    #     pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    #     out, err = pro.communicate()
    #     print(out)
    #     pro.wait()
    #     pro.returncode == 0
    #     assert str(out).find("Error") == -1
    #     assert str(err).find("Error") == -1

    def test_dist_fleet_utils_localfs(self):
        """test_dist_fleet_utils_localfs"""
        cmd = "python -m paddle.distributed.launch --devices 0 --job_id localfs dist_fleet_utils_localfs.py"
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        print(out)
        pro.wait()
        pro.returncode == 0
        assert str(out).find("Error") == -1
        assert str(err).find("Error") == -1

    def test_dist_fleet_utils_recompute(self):
        """test_dist_fleet_utils_recompute"""
        cmd = "python -m paddle.distributed.launch --devices 0 --job_id utils_recompute dist_fleet_utils_recompute.py"
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        print(out)
        pro.wait()
        pro.returncode == 0
        assert str(out).find("Error") == -1
        assert str(err).find("Error") == -1
        assert str(out).find("ABORT") == -1
