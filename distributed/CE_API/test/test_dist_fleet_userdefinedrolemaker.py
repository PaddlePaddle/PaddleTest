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
  * @file test_dist_fleet_userdefinedrolemaker.py
  * @author liujie44@baidu.com
  * @date 2021-11-12 19:01
  * @brief
  *
  **************************************************************************/
"""
import os
import subprocess


os.system("export CUDA_VISIBLE_DEVICES=0,1")


class TestApi(object):
    """test api"""

    def test_dist_fleet_userdefinedrolemaker(self):
        """test_dist_fleet_userdefinedrolemaker"""
        cmd = "python -m paddle.distributed.launch --devices 0,1 --job_id userdefinedrolemake \
        dist_fleet_userdefinedrolemaker.py"
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        print(out)
        pro.wait()
        pro.returncode == 0
        assert str(out).find("Error") == -1
        assert str(err).find("Error") == -1
