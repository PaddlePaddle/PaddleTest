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
  * @file dist_fetch.py
  * @author lvkunpeng@baidu.com
  * @date 2025-07-09
  * @brief
  *
  **************************************************************************/
"""
import pytest
import paddle
from paddle.static import Program, program_guard
from paddle.distributed.auto_parallel.interface import fetch, CollectionNames, add_to_collection, get_collection

from utils import run_priority

paddle.enable_static()

@run_priority(level="P0")
def test_fetch_with_variable():
    """test fetch with variable"""
    with program_guard(paddle.static.Program(), paddle.static.Program()):
        x = paddle.static.data(name="x", shape=[1], dtype="float32")
        fetch(x, name="test_var")
        fetches = get_collection(CollectionNames.FETCHES)
        fetch_var_names = [v for _, v in fetches]
        assert x.name in fetch_var_names


@run_priority(level="P0")
def test_fetch_with_str():
    """test fetch with str"""
    var_name = "x"
    fetch(var_name, name="test_str")
    fetches = get_collection(CollectionNames.FETCHES)
    fetch_var_names = [v for _, v in fetches]
    assert var_name in fetch_var_names



@run_priority(level="P0")
def test_fetch_with_logging():
    with program_guard(paddle.static.Program(), paddle.static.Program()):
        x = paddle.static.data(name="x", shape=[1], dtype="float32")
        fetch(x, name="test_log", logging=True)
        log_collection = get_collection(CollectionNames.LOGGING)
        fetch_var_names = [v for _, v in log_collection]
        assert x.name in fetch_var_names

@run_priority(level="P0")
def test_fetch_type_error():
    with pytest.raises(TypeError):
        fetch(12345)


if __name__ == "__main__":
    test_fetch_with_variable()
    test_fetch_with_str()
    test_fetch_with_logging()
    test_fetch_type_error()

