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
  * @file dist_set_and_get_mesh.py
  * @author lvkunpeng@baidu.com
  * @date 2025-07-09
  * @brief
  *
  **************************************************************************/
"""
import paddle
from paddle.distributed.auto_parallel import get_mesh, set_mesh
from paddle.distributed import ProcessMesh

from utils import run_priority


@run_priority(level="P0")
def test_set_and_get_mesh():
    """test_set_and_get_mesh"""
    mesh = ProcessMesh([[0, 1]], dim_names=["dp", "mp"])

    set_mesh(mesh)
    got = get_mesh()
    rank = paddle.distributed.get_rank()

    assert isinstance(got, ProcessMesh)
    assert rank in got.process_ids

    print("test_set_and_get_mesh ... ok")
    

if __name__ == '__main__':
    test_get_mesh_basic()