#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
/***************************************************************************
  *
  * Copyright (c) 2022 Baidu.com, Inc. All Rights Reserved
  * @file test_dist_MoE_api.py
  * @author liujie44@baidu.com
  * @date 2022-04-24 11:10
  * @brief
  *
  **************************************************************************/
"""
import os
import subprocess

from utils import run_priority

os.system("export CUDA_VISIBLE_DEVICES=0,1")


class TestDistMoEApi(object):
    """TestDistMoEApi"""

    def test_MoE_number_count(self):
        """test_MoE_number_count"""
        cmd = "python dist_MoE_number_count.py"
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        print(out)
        pro.wait()
        pro.returncode == 0
        assert str(out).find("Error") == -1
        assert str(err).find("Error") == -1

    def test_MoE_assign_pos(self):
        """test_MoE_assign_pos"""
        cmd = "python assign_pos dist_MoE_assign_pos.py"
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        print(out)
        pro.wait()
        pro.returncode == 0
        assert str(out).find("Error") == -1
        assert str(err).find("Error") == -1

    def test_MoE_limit_by_capacity(self):
        """test_MoE_limit_by_capacity"""
        cmd = "python dist_MoE_limit_by_capacity.py"
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        print(out)
        pro.wait()
        pro.returncode == 0
        assert str(out).find("Error") == -1
        assert str(err).find("Error") == -1

    def test_MoE_prune_gate_by_capacity(self):
        """test_MoE_prune_gate_by_capacity"""
        cmd = "python dist_MoE_prune_gate_by_capacity.py"
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        print(out)
        pro.wait()
        pro.returncode == 0
        assert str(out).find("Error") == -1
        assert str(err).find("Error") == -1

    def test_MoE_random_routing(self):
        """test_MoE_random_routing"""
        cmd = "python dist_MoE_random_routing.py"
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        print(out)
        pro.wait()
        pro.returncode == 0
        assert str(out).find("Error") == -1
        assert str(err).find("Error") == -1
