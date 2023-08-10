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
  * @file test_dist_fleet_worker.py
  * @author liujie44@baidu.com
  * @date 2021-11-19 15:01
  * @brief
  *
  **************************************************************************/
"""
import os
import subprocess


class TestFleetWorkerServerApi:
    """TestFleetWorkerServerApi"""

    def test_dist_fleet_is_first_worker(self):
        """test_dist_fleet_is_first_worker"""
        cmd = "python -m paddle.distributed.launch --devices 0 --job_id is_first_worker dist_fleet_is_first_worker.py"
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        pro.wait()
        pro.returncode == 0
        assert str(out).find("Error") == -1
        assert str(err).find("Error") == -1

    def test_dist_fleet_worker_index(self):
        """test_dist_fleet_worker_index"""
        cmd = "python -m paddle.distributed.launch --devices 0 --job_id worker_index dist_fleet_worker_index.py"
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        pro.wait()
        pro.returncode == 0
        assert str(out).find("Error") == -1
        assert str(err).find("Error") == -1

    def test_dist_fleet_worker_num(self):
        """test_dist_fleet_worker_num"""
        cmd = "python -m paddle.distributed.launch --devices 0,1 --job_id worker_num dist_fleet_worker_num.py"
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        pro.wait()
        pro.returncode == 0
        assert str(out).find("Error") == -1
        assert str(err).find("Error") == -1

    def test_dist_fleet_is_worker(self):
        """test_dist_fleet_is_worker"""
        cmd = "python -m paddle.distributed.launch --devices 0,1 --job_id is_worker dist_fleet_is_worker.py"
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        pro.wait()
        pro.returncode == 0
        assert str(out).find("Error") == -1
        assert str(err).find("Error") == -1

    def test_dist_fleet_worker_endpoints(self):
        """test_dist_fleet_worker_endpoints"""
        cmd = "python -m paddle.distributed.launch --devices 0,1 --job_id worker_endpoints \
        dist_fleet_worker_endpoints.py"
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        pro.wait()
        pro.returncode == 0
        assert str(out).find("Error") == -1
        assert str(err).find("Error") == -1

    def test_dist_fleet_server_num(self):
        """test_dist_fleet_server_num"""
        cmd = "python -m paddle.distributed.launch --server_num=2 --trainer_num=1 --job_id server_num \
        dist_fleet_server_num.py"
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        pro.wait()
        pro.returncode == 0
        assert str(out).find("Error") == -1
        assert str(err).find("Error") == -1

    def test_dist_fleet_server_index(self):
        """test_dist_fleet_server_endpoints"""
        cmd = "python -m paddle.distributed.launch --server_num=1 --trainer_num=1 --job_id server_index \
        dist_fleet_server_index.py"
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        pro.wait()
        pro.returncode == 0
        assert str(out).find("Error") == -1
        assert str(err).find("Error") == -1

    def test_dist_fleet_server_endpoints(self):
        """test_dist_fleet_server_endpoints"""
        cmd = "python -m paddle.distributed.launch --server_num=2 --trainer_num=1 --job_id server_endpoints \
        dist_fleet_server_endpoints.py"
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        pro.wait()
        pro.returncode == 0
        assert str(out).find("Error") == -1
        assert str(err).find("Error") == -1

    def test_dist_fleet_is_server(self):
        """test_dist_fleet_is_server"""
        cmd = "python -m paddle.distributed.launch --server_num=2 --trainer_num=1 --job_id is_server \
        dist_fleet_is_server.py"
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        pro.wait()
        pro.returncode == 0
        assert str(out).find("Error") == -1
        assert str(err).find("Error") == -1

    def test_dist_fleet_barrier_worker(self):
        """test_dist_fleet_barrier_worker"""
        cmd = "python -m paddle.distributed.launch --devices 0,1 --job_id barrier_worker dist_fleet_barrier_worker.py"
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        pro.wait()
        pro.returncode == 0
        assert str(out).find("Error") == -1
        assert str(err).find("Error") == -1

    def test_dist_fleet_init_worker(self):
        """test_dist_fleet_init_worker"""
        cmd = "python -m paddle.distributed.launch --devices 0 --job_id init_worker dist_fleet_init_worker.py"
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        pro.wait()
        pro.returncode == 0
        assert str(out).find("Error") == -1
        assert str(err).find("Error") == -1

    def test_dist_fleet_init_server(self):
        """test_dist_fleet_init_server"""
        cmd = "python -m paddle.distributed.launch --devices 0 --job_id init_server dist_fleet_init_server.py"
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        pro.wait()
        pro.returncode == 0
        assert str(out).find("Error") == -1
        assert str(err).find("Error") == -1

    def test_dist_fleet_server(self):
        """test_dist_fleet_server"""
        cmd = "python -m paddle.distributed.launch --devices 0 --job_id server dist_fleet_server.py"
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        pro.wait()
        pro.returncode == 0
        assert str(out).find("Error") == -1
        assert str(err).find("Error") == -1

    def test_dist_fleet_worker(self):
        """test_dist_fleet_worker"""
        cmd = "python -m paddle.distributed.launch --devices 0 dist_fleet_worker.py"
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        pro.wait()
        pro.returncode == 0
        assert str(out).find("Error") == -1
        assert str(err).find("Error") == -1
