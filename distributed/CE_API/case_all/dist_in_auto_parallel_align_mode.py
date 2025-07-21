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
  * @file dist_in_auto_parallel_align_mode.py
  * @author lvkunpeng@baidu.com
  * @date 2025-07-08
  * @brief
  *
  **************************************************************************/
"""
import pytest
from unittest.mock import patch
import paddle
import paddle.distributed as dist
from paddle.distributed.auto_parallel.api import in_auto_parallel_align_mode

from utils import run_priority


@run_priority(level="P0")
def test_align_mode_enabled():
    with patch('paddle.base.framework.get_flags', return_value={"FLAGS_enable_auto_parallel_align_mode": True}):
        assert in_auto_parallel_align_mode() is True

@run_priority(level="P0")
def test_align_mode_disabled():
    with patch('paddle.base.framework.get_flags', return_value={"FLAGS_enable_auto_parallel_align_mode": False}):
        assert in_auto_parallel_align_mode() is False

@run_priority(level="P0")
def test_align_mode_flag_missing_raises():
    with patch('paddle.base.framework.get_flags', return_value={}):
        with pytest.raises(KeyError):
            in_auto_parallel_align_mode()



if __name__ == "__main__":
    test_align_mode_enabled()
    test_align_mode_disabled()
    test_align_mode_flag_missing_raises()