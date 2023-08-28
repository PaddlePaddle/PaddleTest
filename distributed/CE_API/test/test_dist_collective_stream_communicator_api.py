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
  * @file test_dist_collective_stream_communicator_api.py
  * @author liujie44@baidu.com
  * @date 2023-11-09 11:00
  * @brief
  *
  **************************************************************************/
"""
import os
import subprocess

from utils import run_priority

os.system("export CUDA_VISIBLE_DEVICES=0,1")


class TestApi(object):
    """test all collective stream communicator api"""

    def test_collective_stream_all_gather(self):
        """test_collective_stream_all_gather"""
        cmd = "python -m paddle.distributed.launch --devices 0,1 \
                --job_id stream_all_gather dist_collective_stream_all_gather.py"
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        print(out)
        pro.wait()
        pro.returncode == 0
        assert str(out).find("Error") == -1
        assert str(err).find("Error") == -1

    def test_collective_stream_all_reduce(self):
        """test_collective_stream_all_reduce"""
        cmd = "python -m paddle.distributed.launch --devices 0,1 \
                --job_id stream_all_reduce dist_collective_stream_all_reduce.py"
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        print(out)
        pro.wait()
        pro.returncode == 0
        assert str(out).find("Error") == -1
        assert str(err).find("Error") == -1

    def test_collective_stream_alltoall_single(self):
        """test_collective_stream_alltoall_single"""
        cmd = "python -m paddle.distributed.launch --devices 0,1 --job_id stream_alltoall_single \
            dist_collective_stream_alltoall_single.py"
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        print(out)
        pro.wait()
        pro.returncode == 0
        assert str(out).find("Error") == -1
        assert str(err).find("Error") == -1

    def test_collective_stream_broadcast(self):
        """test_collective_stream_broadcast"""
        cmd = "python -m paddle.distributed.launch --devices 0,1 \
                --job_id stream_broadcast dist_collective_stream_broadcast.py"
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        print(out)
        pro.wait()
        pro.returncode == 0
        assert str(out).find("Error") == -1
        assert str(err).find("Error") == -1

    def test_collective_stream_reduce_scatter(self):
        """test_collective_stream_reduce_scatter"""
        cmd = "python -m paddle.distributed.launch --devices 0,1 --job_id stream_reduce_scatter \
            dist_collective_stream_reduce_scatter.py"
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        print(out)
        pro.wait()
        pro.returncode == 0
        assert str(out).find("Error") == -1
        assert str(err).find("Error") == -1

    def test_collective_stream_reduce(self):
        """test_collective_stream_reduce"""
        cmd = (
            "python -m paddle.distributed.launch --devices 0,1 --job_id stream_reduce dist_collective_stream_reduce.py"
        )
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        print(out)
        pro.wait()
        pro.returncode == 0
        assert str(out).find("Error") == -1
        assert str(err).find("Error") == -1

    def test_collective_stream_scatter(self):
        """test_collective_stream_scatter"""
        cmd = "python -m paddle.distributed.launch --devices 0,1 \
                --job_id stream_scatter dist_collective_stream_scatter.py"
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        print(out)
        pro.wait()
        pro.returncode == 0
        assert str(out).find("Error") == -1
        assert str(err).find("Error") == -1

    def test_collective_stream_send_recv(self):
        """test_collective_stream_send_recv"""
        cmd = (
            "python -m paddle.distributed.launch --devices 0,1 --job_id stream_send_recv dist_collective_stream_send.py"
        )
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        print(out)
        pro.wait()
        pro.returncode == 0
        assert str(out).find("Error") == -1
        assert str(err).find("Error") == -1
