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
  * @file dist_register_distributed_operator_impl.py
  * @author lvkunpeng@baidu.com
  * @date 2025-07-10
  * @brief
  *
  **************************************************************************/
"""
import pytest
from unittest.mock import MagicMock, patch
import paddle
import paddle.distributed.auto_parallel.static.operators.common as common
from paddle.distributed.auto_parallel.static.operators import register_distributed_operator_impl
from utils import run_priority


def get_distributed_operator_impl_container(op_type):
    """get_distributed_operator_impl_container"""
    return None

@run_priority(level="P0")
def test_register_distributed_operator_impl():
    """test_register_distributed_operator_impl"""
    op_type = "matmul"
    dist_impl = MagicMock()
    mock_container = MagicMock()

    with patch.object(common, "get_distributed_operator_impl_container") as mock_get:
        mock_get.return_value = mock_container
        register_distributed_operator_impl(op_type, dist_impl)

        assert dist_impl.type == op_type
        mock_container.register_impl.assert_called_once_with(dist_impl)
    print("test_register_distributed_operator_impl ... ok")

@run_priority(level="P0")
def test_register_distributed_operator_impl_no_container():
    """test_register_distributed_operator_impl_no_container"""
    op_type = "not_registered_op"
    dist_impl = MagicMock()

    with patch.object(common, "get_distributed_operator_impl_container") as mock_get:
        mock_get.return_value = None

        with pytest.raises(AssertionError, match="Must register distributed operator registry first"):
            register_distributed_operator_impl(op_type, dist_impl)
    print("test_register_distributed_operator_impl_no_container ... ok")


if __name__ == "__main__":
    test_register_distributed_operator_impl()
    test_register_distributed_operator_impl_no_container()