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
  * @file dist_get_log_level_name.py
  * @author lvkunpeng@baidu.com
  * @date 2025-07-18
  * @brief
  *
  **************************************************************************/
"""
import paddle
import logging
import paddle.distributed.fleet.utils.log_util as log_util
from unittest import mock
from utils import run_priority


@run_priority(level="P0")
def test_get_log_level_name():
    """test_get_log_level_name"""
    with mock.patch.object(log_util, "get_log_level_code") as mock_get_code:
        mock_get_code.return_value = logging.DEBUG
        name = log_util.get_log_level_name()
        mock_get_code.assert_called_once()
        assert name == "DEBUG"

        mock_get_code.return_value = logging.ERROR
        name = log_util.get_log_level_name()
        assert name == "ERROR"

        mock_get_code.return_value = 999
        name = log_util.get_log_level_name()
        assert name == "Level 999"
    print("test_get_log_level_name ... ok")


if __name__ == "__main__":
    test_get_log_level_name()


