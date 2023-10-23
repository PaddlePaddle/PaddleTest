#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
# ======================================================================
#
# Copyright (c) 2022 Baidu.com, Inc. All Rights Reserved
#
# ======================================================================
"""
/***************************************************************************
  *
  * Copyright (c) 2022 Baidu.com, Inc. All Rights Reserved
  * @file dist_auto_converter.py
  * @author liujie44@baidu.com
  * @date 2022-04-15 11:00
  * @brief
  *
  **************************************************************************/
"""
import numpy as np
import paddle
from paddle.distributed.auto_parallel.static.converter import Converter
from utils import run_priority


# cur_rank
rank_id = paddle.distributed.get_rank()
# data
complete_tensor = np.arange(512).reshape([8, 8, 8])
tensor_batch = np.split(complete_tensor, 2, axis=0)
tensor_row = np.split(complete_tensor, 2, axis=1)
tensor_col = np.split(complete_tensor, 2, axis=2)
# strategy
name = "tmp_0"
complete_strategy = {name: {"process_shape": [2], "process_group": [0, 1], "dims_mapping": [-1, -1, -1]}}
batch_strategy = {name: {"process_shape": [2], "process_group": [0, 1], "dims_mapping": [0, -1, -1]}}
row_strategy = {"tmp_0": {"process_shape": [2], "process_group": [0, 1], "dims_mapping": [-1, 0, -1]}}
col_strategy = {name: {"process_shape": [2], "process_group": [0, 1], "dims_mapping": [-1, -1, 0]}}


@run_priority(level="P0")
def test_auto_converter_merge_RowToComplete():
    """test auto_converter merge: row_strategy --> complete_strategy"""
    tensor_dict = {name: tensor_row}
    converter = Converter(tensor_dict, row_strategy, complete_strategy)
    converted_tensor_dict = converter.convert()
    assert np.equal(converted_tensor_dict[name], complete_tensor).all()
    print("test auto_converter merge: row_strategy --> complete_strategy ok!!!!!!!")


@run_priority(level="P0")
def test_auto_converter_merge_ColToComplete():
    """test auto_converter merge: col_strategy --> complete_strategy"""
    tensor_dict = {name: tensor_col}
    converter = Converter(tensor_dict, col_strategy, complete_strategy)
    converted_tensor_dict = converter.convert()
    assert np.equal(converted_tensor_dict[name], complete_tensor).all()
    print("test auto_converter merge: col_strategy --> complete_strategy ok!!!!!!!")


@run_priority(level="P0")
def test_auto_converter_merge_BatchToComplete():
    """test auto_converter merge: batch_strategy --> complete_strategy"""
    tensor_dict = {name: tensor_batch}
    converter = Converter(tensor_dict, batch_strategy, complete_strategy)
    converted_tensor_dict = converter.convert()
    assert np.equal(converted_tensor_dict[name], complete_tensor).all()
    print("test auto_converter merge: batch_strategy --> complete_strategy ok!!!!!!!")


@run_priority(level="P0")
def test_auto_converter_slice_CompleteToRow():
    """test auto_converter slice: complete_strategy --> row_strategy"""
    tensor_dict = {name: [complete_tensor]}
    converter = Converter(tensor_dict, complete_strategy, row_strategy)
    converted_tensor_dict = converter.convert()
    assert np.equal(converted_tensor_dict[name], tensor_row[rank_id]).all()
    print("test auto_converter slice: complete_strategy --> row_strategy ok!!!!!!!")


@run_priority(level="P0")
def test_auto_converter_slice_CompleteToCol():
    """test auto_converter slice: complete_strategy --> col_strategy"""
    tensor_dict = {name: [complete_tensor]}
    converter = Converter(tensor_dict, complete_strategy, col_strategy)
    converted_tensor_dict = converter.convert()
    assert np.equal(converted_tensor_dict[name], tensor_col[rank_id]).all()
    print("test auto_converter slice: complete_strategy --> col_strategy ok!!!!!!!")


@run_priority(level="P0")
def test_auto_converter_slice_CompleteToBatch():
    """test auto_converter slice: complete_strategy --> batch_strategy"""
    tensor_dict = {name: [complete_tensor]}
    converter = Converter(tensor_dict, complete_strategy, batch_strategy)
    converted_tensor_dict = converter.convert()
    assert np.equal(converted_tensor_dict[name], tensor_batch[rank_id]).all()
    print("test auto_converter slice: complete_strategy --> batch_strategy ok!!!!!!!")


@run_priority(level="P0")
def test_auto_converter_MergeAndSlice_ColToRow():
    """test auto_converter merge and slice: col_strategy --> row_strategy"""
    tensor_dict = {name: tensor_col}
    converter = Converter(tensor_dict, col_strategy, row_strategy)
    converted_tensor_dict = converter.convert()
    assert np.equal(converted_tensor_dict[name], tensor_row[rank_id]).all()
    print("test auto_converter merge and slice: col_strategy --> row_strategy ok!!!!!!!")


@run_priority(level="P0")
def test_auto_converter_MergeAndSlice_BatchToRow():
    """test auto_converter merge and slice: batch_strategy --> row_strategy"""
    tensor_dict = {name: tensor_batch}
    converter = Converter(tensor_dict, batch_strategy, row_strategy)
    converted_tensor_dict = converter.convert()
    assert np.equal(converted_tensor_dict[name], tensor_row[rank_id]).all()
    print("test auto_converter merge and slice: batch_strategy --> row_strategy ok!!!!!!!")


@run_priority(level="P0")
def test_auto_converter_MergeAndSlice_BatchToCol():
    """test auto_converter merge and slice with prefix match: batch_strategy --> col_strategy"""
    tensor_dict = {name: tensor_batch}
    new_name = "tmp_1"
    col_strategy = {new_name: {"process_shape": [2], "process_group": [0, 1], "dims_mapping": [-1, -1, 0]}}
    converter = Converter(tensor_dict, batch_strategy, col_strategy)
    converted_tensor_dict = converter.convert(strict=False)
    assert np.equal(converted_tensor_dict[new_name], tensor_col[rank_id]).all()
    print("test auto_converter merge and slicewith prefix match: batch_strategy --> col_strategy ok!!!!!!!")


if __name__ == "__main__":
    test_auto_converter_merge_RowToComplete()
    test_auto_converter_merge_ColToComplete()
    test_auto_converter_merge_BatchToComplete()
    test_auto_converter_slice_CompleteToRow()
    test_auto_converter_slice_CompleteToCol()
    test_auto_converter_slice_CompleteToBatch()
    test_auto_converter_MergeAndSlice_ColToRow()
    test_auto_converter_MergeAndSlice_BatchToRow()
    test_auto_converter_MergeAndSlice_BatchToCol()
