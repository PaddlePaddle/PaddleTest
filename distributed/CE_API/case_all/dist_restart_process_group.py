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
  * @file dist_restart_process_group.py
  * @author lvkunpeng@baidu.com
  * @date 2025-07-11
  * @brief
  *
  **************************************************************************/
"""
import paddle
from unittest.mock import MagicMock, patch
import paddle.distributed.collective as collective
from paddle.distributed import restart_process_group
from utils import run_priority


@run_priority(level="P0")
def test_restart_process_group_all():
    """test_restart_process_group_all"""
    mock_pg1 = MagicMock()
    mock_pg2 = MagicMock()
    mock_group1 = MagicMock(process_group=mock_pg1)
    mock_group2 = MagicMock(process_group=mock_pg2)
    
    fake_map = {
        "group1": mock_group1,
        "group2": mock_group2,
    }

    with patch.object(collective, "_get_shutdown_group_map_by_name", return_value=fake_map), \
         patch.object(collective, "_clear_shutdown_group_map_by_name") as mock_clear:

        restart_process_group()

        mock_pg1.restart.assert_called_once()
        mock_pg2.restart.assert_called_once()
        mock_clear.assert_called_once()
    print("test_restart_process_group_all ... ok")

@run_priority(level="P0")
def test_restart_process_group_single():
    """test_restart_process_group_single"""
    mock_pg = MagicMock()
    mock_group = MagicMock(process_group=mock_pg)
    mock_group.name = "group1"

    fake_map = {"group1": mock_group}

    with patch.object(collective, "_get_shutdown_group_map_by_name", return_value=fake_map), \
         patch.object(collective, "_delete_shutdown_group_map_by_name") as mock_delete:

        restart_process_group(group=mock_group)

        mock_pg.restart.assert_called_once()
        mock_delete.assert_called_once_with("group1")
    print("test_restart_process_group_single ... ok")


if __name__ == "__main__":
    test_restart_process_group_all()
    test_restart_process_group_single()