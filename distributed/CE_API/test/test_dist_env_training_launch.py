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
  * @file test_dist_env_training_launch.py
  * @author liujie44@baidu.com
  * @date 2021-11-10 11:10
  * @brief
  *
  **************************************************************************/
"""
from __future__ import print_function
import time
import signal

# import nose.tools as tools
import os
import subprocess

single_data = [0.70575, 0.69835, 0.69342, 0.690098, 0.687781]

all_args = [
    "--devices=0,1 --log_dir=mylog --job_id np2 dist_train_launch.py",
    "--devices=0 --log_dir=mylog --job_id np0 dist_train_launch.py",
    "--devices=1 --log_dir=mylog --job_id np1 dist_train_launch.py",
    "--log_dir=mylog dist_train_launch.py",
    "--devices=0,1 --job_id np2 dist_train_launch.py",
    "--devices=0 --job_id np0 dist_train_launch.py",
    "--devices=1 --job_id np1 dist_train_launch.py",
    "dist_train_launch.py",
]


class TestDistLaunch(object):
    """Test paddle.distributed.launch module cases."""

    def start_proc(self, cmd):
        """start process."""
        pro = subprocess.Popen(
            "python -m paddle.distributed.launch " + cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        out, err = pro.communicate()
        print(out)
        pro.wait()
        pro.returncode == 0
        loss = str(out).split("n")[-2].lstrip("[").rstrip("]\\").split(", ")
        return loss

    def get_result(self, args):
        """get result"""
        test_info1 = []
        test_info2 = []
        loss = self.start_proc(args)
        assert len(loss) == 10

    def test_dist_launch_Collective_2gpus_Tlog(self):
        """test_dist_launch_Collective_2gpus_Tlog."""
        args = all_args[0]
        self.get_result(args)

    def test_dist_launch_Collective_1gpus0_Tlog(self):
        """test_dist_launch_Collective_1gpus_Tlog."""
        args = all_args[1]
        self.get_result(args)

    def test_dist_launch_Collective_1gpus1_Tlog(self):
        """test_dist_launch_Collective_1gpus_Tlog."""
        args = all_args[2]
        self.get_result(args)

    def test_dist_launch_Collective_gpus_default_Tlog(self):
        """test_dist_launch_Collective_gpus_default_Tlog."""
        args = all_args[3]
        self.get_result(args)

    def test_dist_launch_Collective_2gpus(self):
        """test_dist_launch_Collective_2gpus."""
        args = all_args[4]
        self.get_result(args)

    def test_dist_launch_Collective_1gpus0(self):
        """test_dist_launch_Collective_1gpus0."""
        args = all_args[5]
        self.get_result(args)

    def test_dist_launch_Collective_1gpus1(self):
        """test_dist_launch_Collective_1gpus1."""
        args = all_args[6]
        self.get_result(args)

    def test_dist_launch_Collective_gpus_default(self):
        """test_dist_launch_Collective_gpus_default."""
        args = all_args[7]
        self.get_result(args)
