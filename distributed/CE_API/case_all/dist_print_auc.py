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
  * @file dist_print_auc.py
  * @author lvkunpeng@baidu.com
  * @date 2025-07-22
  * @brief
  *
  **************************************************************************/
"""
import paddle
from unittest import mock
import paddle.distributed.metric.metrics as metrics
from utils import run_priority


@run_priority(level="P0")
def test_print_auc_all_phase_pass():
    """test_print_auc_all_phase_pass"""
    with mock.patch.object(metrics, "print_metric") as mock_print_metric:
        mock_print_metric.side_effect = lambda metric_ptr, name: f"printed_{name}"
        metric_ptr = mock.Mock()
        metric_ptr.get_metric_name_list.return_value = [
            "pass_train_auc",
            "pass_join_auc",
            "day_train_auc"
        ]

        result = metrics.print_auc(metric_ptr, is_day=False, phase="all")

        assert result == ["printed_pass_train_auc", "printed_pass_join_auc"]
        metric_ptr.get_metric_name_list.assert_called_once_with(0)
        print("test_print_auc_all_phase_pass ... ok")

@run_priority(level="P0")
def test_print_auc_join_phase_day():
    """test_print_auc_join_phase_day"""
    with mock.patch.object(metrics, "print_metric") as mock_print_metric:
        mock_print_metric.side_effect = lambda metric_ptr, name: f"printed_{name}"

        metric_ptr = mock.Mock()
        metric_ptr.get_metric_name_list.return_value = [
            "day_join_auc",
            "day_train_auc",
            "pass_join_auc"
        ]

        result = metrics.print_auc(metric_ptr, is_day=True, phase="join")

        assert result == ["printed_day_join_auc"]
        metric_ptr.get_metric_name_list.assert_called_once_with(-1)
        print("test_print_auc_join_phase_day ... ok")

@run_priority(level="P0")
def test_print_auc_train_phase_pass():
    """test_print_auc_train_phase_pass"""
    with mock.patch.object(metrics, "print_metric") as mock_print_metric:
        mock_print_metric.side_effect = lambda metric_ptr, name: f"printed_{name}"

        metric_ptr = mock.Mock()
        metric_ptr.get_metric_name_list.return_value = [
            "pass_train_auc",
            "pass_join_auc",
            "day_train_auc"
        ]

        result = metrics.print_auc(metric_ptr, is_day=False, phase="train")

        assert result == ["printed_pass_train_auc"]
        metric_ptr.get_metric_name_list.assert_called_once_with(0)
        print("test_print_auc_train_phase_pass ... ok")


if __name__ == "__main__":
    test_print_auc_all_phase_pass()
    test_print_auc_join_phase_day()
    test_print_auc_train_phase_pass()