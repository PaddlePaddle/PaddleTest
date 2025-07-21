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
  * @file dist_register_distributed_operator_impl_container.py
  * @author lvkunpeng@baidu.com
  * @date 2025-07-11
  * @brief
  *
  **************************************************************************/
"""
import paddle
from unittest.mock import MagicMock
import paddle.distributed.auto_parallel.static.operators.common as common
from paddle.distributed.auto_parallel.static.operators import register_distributed_operator_impl_container
from utils import run_priority


@run_priority(level="P0")
def test_register_distributed_operator_impl_container():
    """test_register_distributed_operator_impl_container"""
    old_registry = common._g_distributed_operator_impl_containers.copy()

    try:
        common._g_distributed_operator_impl_containers.clear()
        mock_container = MagicMock()
        mock_container.type = "matmul"
        register_distributed_operator_impl_container(mock_container)

        result = common.get_distributed_operator_impl_container("matmul")
        assert result is mock_container
        print("test_register_distributed_operator_impl_container ... ok")
    finally:
        common._g_distributed_operator_impl_containers = old_registry


if __name__ == '__main__':
    test_register_distributed_operator_impl_container()