#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
/***************************************************************************
  *
  * Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
  * @file test_rpc.py
  * @author liujie44@baidu.com
  * @date 2023-08-22 11:10
  * @brief
  *
  **************************************************************************/
"""
import os
import subprocess
from utils import run_priority

os.system("export CUDA_VISIBLE_DEVICES=0,1")


class TestDistRpcApi(object):
    """TestDistRpcApi"""

    def test_rpc_init(self):
        """test_rpc_init"""
        cmd = "python -m paddle.distributed.launch --devices 0,1 --job_id rpc_init dist_rpc_init.py"
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        print(out)
        pro.wait()
        pro.returncode == 0
        assert str(out).find("Error") == -1
        assert str(err).find("Error") == -1

    def test_rpc_sync(self):
        """test_rpc_sync"""
        cmd = "python -m paddle.distributed.launch --devices 0,1 --job_id rpc_sync dist_rpc_sync.py"
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        print(out)
        pro.wait()
        pro.returncode == 0
        assert str(out).find("Error") == -1
        assert str(err).find("Error") == -1

    def test_rpc_async(self):
        """test_rpc_async"""
        cmd = "python -m paddle.distributed.launch --devices 0,1 --job_id rpc_async dist_rpc_async.py"
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        print(out)
        pro.wait()
        pro.returncode == 0
        assert str(out).find("Error") == -1
        assert str(err).find("Error") == -1

    def test_rpc_get_worker_info(self):
        """test_rpc_get_worker_info"""
        cmd = "python -m paddle.distributed.launch --devices 0,1 \
                --job_id rpc_get_worker_info dist_rpc_get_worker_info.py"
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        print(out)
        pro.wait()
        pro.returncode == 0
        assert str(out).find("Error") == -1
        assert str(err).find("Error") == -1

    def test_rpc_get_all_worker_infos(self):
        """test_rpc_get_all_worker_infos"""
        cmd = "python -m paddle.distributed.launch --devices 0,1 \
                --job_id rpc_get_all_worker_infos dist_rpc_get_all_worker_infos.py"
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        print(out)
        pro.wait()
        pro.returncode == 0
        assert str(out).find("Error") == -1
        assert str(err).find("Error") == -1

    def test_rpc_get_current_worker_info(self):
        """test_rpc_get_current_worker_info"""
        cmd = "python -m paddle.distributed.launch --devices 0,1 \
                --job_id rpc_get_current_worker_info dist_rpc_get_current_worker_info.py"
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        print(out)
        pro.wait()
        pro.returncode == 0
        assert str(out).find("Error") == -1
        assert str(err).find("Error") == -1
