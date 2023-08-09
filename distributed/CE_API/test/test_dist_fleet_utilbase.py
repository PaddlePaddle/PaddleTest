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
  * @file test_dist_fleet_utilbase.py
  * @author liujie44@baidu.com
  * @date 2021-11-22 19:16
  * @brief
  *
  **************************************************************************/
"""
import os
import subprocess


class TestFleetUtilBaseApi(object):
    """test all api"""

    def test_dist_fleet_utilbase_all_reduce(self):
        """test_dist_fleet_utilbase_all_reduce"""
        cmd = "python -m paddle.distributed.launch --server_num 2 --trainer_num 1 --job_id utilbase_all_reduce \
        dist_fleet_utilbase_all_reduce.py"
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        print(out)
        pro.wait()
        pro.returncode == 0
        assert str(out).find("Error") == -1
        assert str(err).find("Error") == -1

    def test_dist_fleet_utilbase_barrier(self):
        """test_dist_fleet_utilbase_barrier"""
        cmd = "python -m paddle.distributed.launch --server_num 2 --trainer_num 1 --job_id utilbase_barrier \
        dist_fleet_utilbase_barrier.py"
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        print(out)
        pro.wait()
        pro.returncode == 0
        assert str(out).find("Error") == -1
        assert str(err).find("Error") == -1

    def test_dist_fleet_utilbase_all_gather(self):
        """test_dist_fleet_utilbase_all_gather"""
        cmd = "python -m paddle.distributed.launch --server_num 1 --trainer_num 1 --job_id utilbase_all_gather \
        dist_fleet_utilbase_all_gather.py"
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        print(out)
        pro.wait()
        pro.returncode == 0
        assert str(out).find("Error") == -1
        assert str(err).find("Error") == -1

    def test_dist_fleet_utilbase_get_file_shard(self):
        """test_dist_fleet_utilbase_get_file_shard"""
        cmd = "python -m paddle.distributed.launch --server_num 2 --trainer_num 1 --job_id utilbase_get_file_shard \
        dist_fleet_utilbase_get_file_shard.py"
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        print(out)
        pro.wait()
        pro.returncode == 0
        assert str(out).find("Error") == -1
        assert str(err).find("Error") == -1

    def test_dist_fleet_utilbase_print_on_rank(self):
        """test_dist_fleet_utilbase_print_on_rank"""
        cmd = "python -m paddle.distributed.launch --server_num 2 --trainer_num 1 --job_id utilbase_print_on_rank \
        dist_fleet_utilbase_print_on_rank.py"
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        print(out)
        pro.wait()
        pro.returncode == 0
        assert str(out).find("Error") == -1
        assert str(err).find("Error") == -1
