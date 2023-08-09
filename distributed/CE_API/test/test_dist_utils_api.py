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
  * @file test_dist_utils_api.py
  * @author liujie44@baidu.com
  * @date 2021-11-03 19:57
  * @brief
  *
  **************************************************************************/
"""
import os
import sys
import subprocess
from utils import run_priority

os.system("export CUDA_VISIBLE_DEVICES=0,1")


class TestUtilsAPI(object):
    """TestUtilsAPI"""

    def test_dist_utils_global_gather(self):
        """test_dist_utils_global_gather"""
        cmd = "python -m paddle.distributed.launch --devices=0,1 --job_id utils_global_gather \
        dist_utils_global_gather.py"
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        print(out)
        pro.wait()
        pro.returncode == 0
        assert str(out).find("Error") == -1
        assert str(err).find("Error") == -1

    def test_dist_utils_global_scatter(self):
        """test_dist_utils_global_scatter"""
        cmd = "python -m paddle.distributed.launch --devices=0,1 --job_id utils_global_scatter \
        dist_utils_global_scatter.py"
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        print(out)
        pro.wait()
        pro.returncode == 0
        assert str(out).find("Error") == -1
        assert str(err).find("Error") == -1
