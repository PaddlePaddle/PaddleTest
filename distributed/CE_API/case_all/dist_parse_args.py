#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
# ======================================================================
#
# Copyright (c) 2024 Baidu.com, Inc. All Rights Reserved
#
# ======================================================================
"""
/***************************************************************************
  *
  * Copyright (c) 2025 Baidu.com, Inc. All Rights Reserved
  * @file dist_parse_args.py
  * @author lvkunpeng@baidu.com
  * @date 2025-07-21
  * @brief
  *
  **************************************************************************/
"""
import paddle
import sys
import os
from unittest import mock
from paddle.distributed.launch.context import parse_args 
from utils import run_priority


@run_priority(level="P0")
def test_parse_args_base_params():
    """test_parse_args_base_params"""
    fake_args = [
        "launch.py",
        "--master", "127.0.0.1:8080",
        "--legacy", "True",
        "--rank", "2",
        "--log_level", "DEBUG",
        "--nnodes", "2",
        "--nproc_per_node", "4",
        "--training_script", "train.py",
        "arg1", "arg2"
    ]

    with mock.patch.object(sys, 'argv', fake_args):
        with mock.patch.dict(os.environ, {}, clear=True):
            args, unknown = parse_args()

    assert args.master == "127.0.0.1:8080"
    assert args.legacy is True
    assert args.rank == 2
    assert args.log_level == "DEBUG"
    assert args.nnodes == "2"
    assert args.nproc_per_node == 4
    assert args.training_script == "train.py"
    assert args.training_script_args == ["arg1", "arg2"]
    print("test_parse_args_base_params ... ok")

@run_priority(level="P0")
def test_parse_args_ps_params():
    """test_parse_args_ps_params"""
    fake_args = [
        "launch.py",
        "--servers", "server1,server2",
        "--trainers", "trainer1,trainer2",
        "--trainer_num", "10",
        "--server_num", "5",
        "--gloo_port", "8888",
        "--with_gloo", "0",
        "--training_script", "train.py"
    ]

    with mock.patch.object(sys, 'argv', fake_args):
        with mock.patch.dict(os.environ, {}, clear=True):
            args, unknown = parse_args()

    assert args.servers == "server1,server2"
    assert args.trainers == "trainer1,trainer2"
    assert args.trainer_num == 10
    assert args.server_num == 5
    assert args.gloo_port == 8888
    assert args.with_gloo == "0"
    print("test_parse_args_ps_params ... ok")

@run_priority(level="P0")
def test_parse_args_elastic_params():
    """test_parse_args_elastic_params"""
    fake_args = [
        "launch.py",
        "--max_restart", "7",
        "--elastic_level", "1",
        "--elastic_timeout", "60",
        "--training_script", "train.py"
    ]

    with mock.patch.object(sys, 'argv', fake_args):
        with mock.patch.dict(os.environ, {}, clear=True):
            args, unknown = parse_args()

    assert args.max_restart == 7
    assert args.elastic_level == 1
    assert args.elastic_timeout == 60
    print("test_parse_args_elastic_params ... ok")

@run_priority(level="P0")
def test_parse_args_paddle_trainer_id():
    """test_parse_args_paddle_trainer_id"""
    fake_args = [
        "launch.py",
        "--rank", "2",
        "--training_script", "train.py"
    ]

    with mock.patch.object(sys, 'argv', fake_args):
        with mock.patch.dict(os.environ, {'PADDLE_TRAINER_ID': '3'}):
            args, unknown = parse_args()

    assert args.rank == 3
    print("test_parse_args_paddle_trainer_id ... ok")


if __name__ == "__main__":
    test_parse_args_base_params()
    test_parse_args_ps_params()
    test_parse_args_elastic_params()
    test_parse_args_paddle_trainer_id()