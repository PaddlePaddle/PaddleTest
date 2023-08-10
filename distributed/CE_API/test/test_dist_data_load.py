#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
/***************************************************************************
  *
  * Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
  * @file test_dist_data_load.py
  * @author liujie44@baidu.com
  * @date 2021-11-09 11:10
  * @brief
  *
  **************************************************************************/
"""
import os
import subprocess
from utils import run_priority

os.system("export CUDA_VISIBLE_DEVICES=0,1")


class TestDistInmemoryDataSetApi(object):
    """TestDistInmemoryDataSetApi"""

    def test_data_inmemorydataset(self):
        """test_data_inmemorydataset"""
        cmd = "python -m paddle.distributed.launch --devices 0,1 --job_id inmemorydataset dist_data_inmemorydataset.py"
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        print(out)
        pro.wait()
        pro.returncode == 0
        assert str(out).find("Error") == -1
        assert str(err).find("Error") == -1

    def test_data_queuedataset(self):
        """test_data_queuedataset"""
        cmd = "python -m paddle.distributed.launch --devices 0,1 --job_id queuedataset dist_data_queuedataset.py"
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        print(out)
        pro.wait()
        pro.returncode == 0
        assert str(out).find("Error") == -1
        assert str(err).find("Error") == -1
