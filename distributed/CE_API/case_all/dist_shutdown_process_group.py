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
  * @file dist_shutdown_process_group.py
  * @author lvkunpeng@baidu.com
  * @date 2025-07-11
  * @brief
  *
  **************************************************************************/
"""
import paddle
from unittest.mock import MagicMock, patch
import paddle.distributed.collective as collective
from paddle.distributed import shutdown_process_group
from utils import run_priority

@run_priority(level="P0")
def test_shutdown_process_group_all():
    """test_shutdown_process_group_all"""
    pg1 = MagicMock()
    pg2 = MagicMock()
    pg3 = MagicMock()
    group1 = MagicMock(process_group=pg1)
    group2 = MagicMock(process_group=pg2)
    default_group = MagicMock(process_group=pg3)

    group_map = {
        "group1": group1,
        "group2": group2,
        "default": default_group,
    }

    with patch.object(collective, "_get_group_map_by_name", return_value=group_map), \
         patch.object(collective, "_get_shutdown_group_map_by_name", return_value={}), \
         patch.object(collective, "_update_shutdown_group_map_by_name") as mock_update, \
         patch.object(collective, "_default_group_name", "default"):

        shutdown_process_group()

        pg1.shutdown.assert_called_once()
        pg2.shutdown.assert_called_once()
        pg3.shutdown.assert_not_called()
        assert mock_update.call_count == 2

    print("test_shutdown_process_group_all ... ok")

@run_priority(level="P0")
def test_shutdown_process_group_single():
    """test_shutdown_process_group_single"""
    mock_pg = MagicMock()
    group = MagicMock(process_group=mock_pg)
    group.name = "group1"

    with patch.object(collective, "_get_shutdown_group_map_by_name", return_value={}), \
         patch.object(collective, "_update_shutdown_group_map_by_name") as mock_update:

        shutdown_process_group(group=group)

        mock_pg.shutdown.assert_called_once()
        mock_update.assert_called_once_with("group1", group)

    print("test_shutdown_process_group_single ... ok")

@run_priority(level="P0")
def test_shutdown_process_group_already_closed():
    """test_shutdown_process_group_already_closed"""
    mock_pg = MagicMock()
    group = MagicMock(process_group=mock_pg)
    group.name = "group1"

    shutdown_map = {"group1": group}

    with patch.object(collective, "_get_shutdown_group_map_by_name", return_value=shutdown_map), \
         patch.object(collective, "_update_shutdown_group_map_by_name") as mock_update:

        shutdown_process_group(group=group)

        mock_pg.shutdown.assert_not_called()
        mock_update.assert_not_called()

if __name__ == "__main__":
    test_shutdown_process_group_all()
    test_shutdown_process_group_single()
    test_shutdown_process_group_already_closed()