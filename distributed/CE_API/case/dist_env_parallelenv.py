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
  * @file dist_env_parallelenv.py
  * @author liujie44@baidu.com
  * @date 2021-11-09 11:10
  * @brief
  *
  **************************************************************************/
"""
import os
import sys
import paddle
import paddle.distributed as dist
from utils import run_priority

dist.init_parallel_env()
parallel_env = dist.ParallelEnv()


@run_priority(level="P0")
def test_parallelenv_rank():
    """test_parallelenv_rank"""
    parallel_env.rank
    print("The rank is %d" % parallel_env.rank)
    print("{} ... ok".format(sys._getframe().f_code.co_name))


@run_priority(level="P0")
def test_parallelenv_world_size():
    """test_parallelenv_world_size"""
    assert parallel_env.world_size == 2
    print("{} ... ok".format(sys._getframe().f_code.co_name))


@run_priority(level="P0")
def test_parallelenv_device_id():
    """test_parallelenv_device_id"""
    parallel_env.device_id
    print("The device_id is %d" % parallel_env.device_id)
    print("{} ... ok".format(sys._getframe().f_code.co_name))


@run_priority(level="P0")
def test_parallelenv_current_endpoint():
    """test_parallelenv_current_endpoint"""
    print(parallel_env.current_endpoint)
    assert str(parallel_env.current_endpoint).startswith("10.")
    print("{} ... ok".format(sys._getframe().f_code.co_name))


@run_priority(level="P0")
def test_parallelenv_trainer_endpoints():
    """test_parallelenv_trainer_endpoints"""
    print(parallel_env.trainer_endpoints)
    assert len(parallel_env.trainer_endpoints) == 2
    print("{} ... ok".format(sys._getframe().f_code.co_name))


if __name__ == "__main__":
    test_parallelenv_rank()
    test_parallelenv_world_size()
    test_parallelenv_device_id()
    test_parallelenv_current_endpoint()
    test_parallelenv_trainer_endpoints()
