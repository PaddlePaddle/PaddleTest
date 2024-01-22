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
  * @file test_dist_fleet_paddlecloudrolemaker.py
  * @author liujie44@baidu.com
  * @date 2021-11-12 16:16
  * @brief
  *
  **************************************************************************/
"""
import os
import subprocess


os.system("export CUDA_VISIBLE_DEVICES=0,1")


class TestApi(object):
    """test all api"""

    def test_dist_fleet_paddlecloudrolemaker(self):
        """test_dist_fleet_paddlecloudrolemaker"""
        cmd = "python -m paddle.distributed.launch --devices 0,1 --job_id paddlecloudrolemaker \
        dist_fleet_paddlecloudrolemaker.py"
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        print(out)
        pro.wait()
        pro.returncode == 0
        assert str(out).find("Error") == -1
        assert str(err).find("Error") == -1
