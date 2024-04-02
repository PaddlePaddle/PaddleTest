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
  * @file test_dist_fleet_dygraph_api.py.py
  * @author liyang109@baidu.com
  * @date 2020-11-16 14:33
  * @brief
  *
  **************************************************************************/
"""
from __future__ import print_function
import time
import signal
import os
import subprocess


class TestDygraph(object):
    """Test dygraph"""

    def test_dist_fleet_dygraph_api_2gpus(self):
        """test_dist_fleet_dygraph_api_2gpus."""
        cmd = "python -m paddle.distributed.launch --devices 0,1 --job_id dygraph_api_np2 dist_fleet_dygraph_api.py"
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        pro.wait()
        pro.returncode == 0
        assert str(out).find("Error") == -1
        assert str(err).find("Error") == -1

    def test_dist_fleet_dygraph_api_1gpus0(self):
        """test_dist_fleet_dygraph_api_1gpus"""
        cmd = "python -m paddle.distributed.launch --devices 0 --job_id dygraph_api_np0 dist_fleet_dygraph_api.py"
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        pro.wait()
        pro.returncode == 0
        assert str(out).find("Error") == -1
        assert str(err).find("Error") == -1

    # def test_dist_fleet_dygraph_api_1gpus1(self):
    #   """test_dist_fleet_dygraph_api_1gpus"""
    #   cmd='python -m paddle.distributed.launch --devices 1 --job_id dygraph_api_np1 dist_fleet_dygraph_api.py'
    #   pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    #   out, err = pro.communicate()
    #   pro.wait()
    #   pro.returncode == 0
    #   assert str(out).find("Error") == -1
    #   assert str(err).find("Error") == -1

    def test_dist_fleet_dygraph_lr_2gpus(self):
        """test_dist_fleet_dygraph_lr_2gpus"""
        cmd = "python -m paddle.distributed.launch --devices 0,1 --job_id dygraph_lr_np2 dist_fleet_dygraph_lr.py"
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        pro.wait()
        pro.returncode == 0
        assert str(out).find("Error") == -1
        assert str(err).find("Error") == -1

    def test_dist_fleet_dygraph_lr_1gpus0(self):
        """test_dist_fleet_dygraph_lr_1gpus"""
        cmd = "python -m paddle.distributed.launch --devices 0 --job_id dygraph_lr_np0 dist_fleet_dygraph_lr.py"
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        pro.wait()
        pro.returncode == 0
        assert str(out).find("Error") == -1
        assert str(err).find("Error") == -1

    # def test_dist_fleet_dygraph_lr_1gpus1(self):
    #   """test_dist_fleet_dygraph_lr_1gpus"""
    #   cmd='python -m paddle.distributed.launch --devices 1 --job_id dygraph_lr_np1 dist_fleet_dygraph_lr.py'
    #   pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    #   out, err = pro.communicate()
    #   pro.wait()
    #   pro.returncode == 0
    #   assert str(out).find("Error") == -1
    #   assert str(err).find("Error") == -1
