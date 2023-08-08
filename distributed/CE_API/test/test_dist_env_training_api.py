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
  * @file test_dist_env_training_api.py
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


class TestApi(object):
    """test all environment and training api"""

    def test_env_get_rank(self):
        """test_env_get_rank"""
        cmd = "python -m paddle.distributed.launch --devices 0,1 --job_id get_rank dist_env_get_rank.py"
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        print(out)
        pro.wait()
        pro.returncode == 0
        assert str(out).find("Error") == -1
        assert str(err).find("Error") == -1

    def test_env_get_world_size(self):
        """test_env_get_world_size"""
        cmd = "python -m paddle.distributed.launch --devices 0,1 --job_id get_world_size dist_env_get_world_size.py"
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        print(out)
        pro.wait()
        pro.returncode == 0
        assert str(out).find("Error") == -1
        assert str(err).find("Error") == -1

    def test_env_init_parallel_env(self):
        """test_env_init_parallel_env"""
        cmd = "python -m paddle.distributed.launch --devices 0,1 --job_id init_parallel_env \
        dist_env_init_parallel_env.py"
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        print(out)
        pro.wait()
        pro.returncode == 0
        assert str(out).find("Error") == -1
        assert str(err).find("Error") == -1

    def test_train_dist_fleet_spawn(self):
        """test_train_dist_fleet_spawn"""
        cmd = "python dist_train_spawn.py"
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        print(out)
        pro.wait()
        pro.returncode == 0
        assert str(out).find("Error") == -1
        assert str(err).find("Error") == -1

    def test_env_parallelenv(self):
        """test_env_parallelenv"""
        cmd = "python -m paddle.distributed.launch --devices 0,1 --job_id parallelenv dist_env_parallelenv.py"
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        print(out)
        pro.wait()
        pro.returncode == 0
        assert str(out).find("Error") == -1
        assert str(err).find("Error") == -1

    def test_env_is_available(self):
        """test_env_is_available"""
        cmd = "python -m paddle.distributed.launch --devices 0,1 --job_id is_available dist_env_is_available.py"
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        print(out)
        pro.wait()
        pro.returncode == 0
        assert str(out).find("Error") == -1
        assert str(err).find("Error") == -1

    def test_env_destroy_process_group(self):
        """test_env_destroy_process_group"""
        cmd = "python -m paddle.distributed.launch --devices 0,1 --job_id destroy_process_group \
            dist_env_destroy_process_group.py"
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        print(out)
        pro.wait()
        pro.returncode == 0
        assert str(out).find("Error") == -1
        assert str(err).find("Error") == -1

    def test_env_get_backend(self):
        """test_env_get_backend"""
        cmd = "python -m paddle.distributed.launch --devices 0,1 --job_id get_backend dist_env_get_backend.py"
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        print(out)
        pro.wait()
        pro.returncode == 0
        assert str(out).find("Error") == -1
        assert str(err).find("Error") == -1
