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
  * @file dist_set_log_level.py
  * @author lvkunpeng@baidu.com
  * @date 2025-07-18
  * @brief
  *
  **************************************************************************/
"""
import paddle
import logging
import pytest
import paddle.distributed.fleet.utils.log_util as log_util
from unittest import mock
from utils import run_priority


@run_priority(level="P0")
def test_set_log_level_with_int():
    """test_set_log_level_with_int"""
    with mock.patch.object(log_util.logger, "setLevel") as mock_set_level:
        log_util.set_log_level(logging.DEBUG)
        mock_set_level.assert_called_once_with(logging.DEBUG)
    print("test_set_log_level_with_int ... ok")

@run_priority(level="P0")
def test_set_log_level_with_str():
    """test_set_log_level_with_str"""
    with mock.patch.object(log_util.logger, "setLevel") as mock_set_level:
        log_util.set_log_level("info")
        mock_set_level.assert_called_once_with("INFO")
    print("test_set_log_level_with_str ... ok")

@run_priority(level="P0")
def test_set_log_level_invalid_type():
    """test_set_log_level_invalid_type"""
    with pytest.raises(AssertionError, match="level's type must be str or int"):
        log_util.set_log_level(3.14)
    print("test_set_log_level_invalid_type ... ok")

@run_priority(level="P0")
def test_set_log_level_upper_str():
    """test_set_log_level_upper_str"""
    with mock.patch.object(log_util.logger, "setLevel") as mock_set_level:
        log_util.set_log_level("warning")
        mock_set_level.assert_called_once_with("WARNING")
    print("test_set_log_level_upper_str ... ok")


if __name__ == '__main__':
    test_set_log_level_with_int()
    test_set_log_level_with_str()
    test_set_log_level_invalid_type()
    test_set_log_level_upper_str()








