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
  * @file dist_get_log_level_code.py
  * @author lvkunpeng@baidu.com
  * @date 2025-07-18
  * @brief
  *
  **************************************************************************/
"""
import paddle
import logging
from unittest import mock
import paddle.distributed.fleet.utils.log_util as log_util
from utils import run_priority


@run_priority(level="P0")
def test_get_log_level_code():
    with mock.patch.object(log_util.logger, 'getEffectiveLevel') as mock_get_level:
        mock_get_level.return_value = logging.WARNING
        result = log_util.get_log_level_code()
        mock_get_level.assert_called_once()
        assert result == logging.WARNING
    print("test_get_log_level_code ... ok")


if __name__ == '__main__':
    test_get_log_level_code()