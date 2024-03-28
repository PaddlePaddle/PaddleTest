#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
/***************************************************************************
  *
  * Copyright (c) 2024 Baidu.com, Inc. All Rights Reserved
  * @file test_dist_gather.py
  * @author liujie44@baidu.com
  * @date 2024-03-04
  * @brief
  *
  **************************************************************************/
"""
import os
import subprocess
from utils import run_priority

os.system("export CUDA_VISIBLE_DEVICES=0,1")


class TestDistgatherApi(object):
    """TestDistgatherApi"""

    def test_gather(self):
        """test_gather"""
        cmd = "python -m paddle.distributed.launch --devices 0,1 --job_id gather dist_gather.py"
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        print(out)
        pro.wait()
        pro.returncode == 0
        assert str(out).find("Error") == -1
        assert str(err).find("Error") == -1
