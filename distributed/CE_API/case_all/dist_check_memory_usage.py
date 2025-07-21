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
  * @file dist_check_memory_usage.py
  * @author lvkunpeng@baidu.com
  * @date 2025-07-21
  * @brief
  *
  **************************************************************************/
"""
import paddle
import logging
import subprocess
import paddle.distributed.fleet.utils.log_util as log_util
from unittest import mock
from utils import run_priority


@run_priority(level="P0")
def test_check_memory_usage():
    """test_check_memeory_usage"""
    with mock.patch.object(log_util, "logger") as mock_logger, \
         mock.patch.object(log_util.paddle.device, "cuda") as mock_cuda, \
         mock.patch.object(log_util.paddle.device, "cpu", create=True) as mock_cpu, \
         mock.patch.object(log_util.subprocess, "run") as mock_subproc:

        mock_cuda.max_memory_allocated.return_value = 2 * 1024**3
        mock_cuda.max_memory_reserved.return_value = 3 * 1024**3
        mock_cuda.memory_allocated.return_value = 1 * 1024**3
        mock_cuda.memory_reserved.return_value = 1.5 * 1024**3

        mock_cuda.max_pinned_memory_allocated.return_value = 0.5 * 1024**3
        mock_cuda.max_pinned_memory_reserved.return_value = 0.8 * 1024**3
        mock_cuda.pinned_memory_allocated.return_value = 0.3 * 1024**3
        mock_cuda.pinned_memory_reserved.return_value = 0.6 * 1024**3

        mock_cpu.max_memory_allocated.return_value = 0.2 * 1024**3
        mock_cpu.max_memory_reserved.return_value = 0.3 * 1024**3
        mock_cpu.memory_allocated.return_value = 0.1 * 1024**3
        mock_cpu.memory_reserved.return_value = 0.15 * 1024**3

        mock_subproc.return_value.stdout = (
            "              total        used        free      shared  buff/cache   available\n"
            "Mem:           16Gi        6Gi         8Gi        0.5Gi       2Gi         9Gi\n"
            "Swap:           4Gi        1Gi         3Gi\n"
        )

        log_util.check_memory_usage("test_run")
        mock_subproc.assert_called_once_with(["free", "-h"], capture_output=True, text=True)

        logged = [call.args[0] for call in mock_logger.info.call_args_list]
        assert any("checking gpu memory usage test_run" in m for m in logged)
        assert any("checking pinned memory usage test_run" in m for m in logged)
        assert any("checking cpu memory usage test_run" in m for m in logged)
        assert any("Memory - Total" in m for m in logged)
        print(mock_logger.info.call_args_list)
        print("test_check_memory_usage ... ok")


if __name__ == "__main__":
    test_check_memory_usage()