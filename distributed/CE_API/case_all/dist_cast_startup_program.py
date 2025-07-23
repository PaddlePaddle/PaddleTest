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
  * @file dist_cast_startup_program.py
  * @author lvkunpeng@baidu.com
  * @date 2025-07-22
  * @brief
  *
  **************************************************************************/
"""
import paddle
from paddle.framework import core
from unittest import mock
import paddle.distributed.passes.auto_parallel_fp16 as auto_parallel_fp16
from utils import run_priority


@run_priority(level="P0")
def test_cast_startup_program():
    """test_cast_startup_program"""
    auto_parallel_fp16.__target_dtype__ = core.VarDesc.VarType.FP16

    param = mock.Mock()
    param.name = "var_1"
    param.dtype = core.VarDesc.VarType.FP16

    block = mock.Mock()
    block.all_parameters.return_value = [param]

    main_program = mock.Mock()
    main_program.blocks = [block]

    op = mock.Mock()
    op.type = "fill_constant"
    op.output_arg_names = ["var_1"]
    op.input_arg_names = []
    op.has_attr.return_value = True
    op.attr.return_value = core.VarDesc.VarType.FP32
    op._set_attr = mock.Mock()

    var_desc = mock.Mock()
    var_desc.set_dtype = mock.Mock()

    var = mock.Mock()
    var.dtype = paddle.float32
    var.desc = var_desc

    global_block = mock.Mock()
    global_block.ops = [op]
    global_block.var.return_value = var

    startup_program = mock.Mock()
    startup_program.global_block.return_value = global_block

    with mock.patch.object(auto_parallel_fp16, "default_main_program", return_value=main_program), \
         mock.patch.object(auto_parallel_fp16, "default_startup_program", return_value=startup_program):

        auto_parallel_fp16.cast_startup_program()

        block.all_parameters.assert_called_once()
        var_desc.set_dtype.assert_called_once_with(core.VarDesc.VarType.FP16)
        op._set_attr.assert_called_once_with('dtype', core.VarDesc.VarType.FP16)
        print("test_cast_startup_program ... ok")


if __name__ == "__main__":
    test_cast_startup_program()