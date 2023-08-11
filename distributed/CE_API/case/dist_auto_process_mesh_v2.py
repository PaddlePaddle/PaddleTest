#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
# ======================================================================
#
# Copyright (c) 2017 Baidu.com, Inc. All Rights Reserved
#
# ======================================================================
"""
/***************************************************************************
  *
  * Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
  * @file dist_auto_process_mesh_v2.py
  * @author liujie44@baidu.com
  * @date 2021-11-09 11:00
  * @brief
  *
  **************************************************************************/
"""
import sys
import numpy as np
from paddle.distributed.auto_parallel.static.process_mesh_v2 import (
    ProcessMesh,
    compute_compatible_process_mesh,
    merge_process_mesh,
)

from utils import run_priority


@run_priority(level="P0")
def test_auto_process_mesh():
    """test_auto_process_mesh"""
    mesh = [[0, 1, 2], [3, 4, 5]]
    mesh2 = [[0, 1], [2, 3]]
    process_mesh = ProcessMesh(mesh, dim_names=["x", "y"])
    process_mesh2 = ProcessMesh(mesh2)
    assert process_mesh.shape == [2, 3]
    assert process_mesh.process_ids == [0, 1, 2, 3, 4, 5]
    assert process_mesh.dim_names == ["x", "y"]
    assert process_mesh.size == 6
    assert process_mesh.ndim == 2
    assert process_mesh.dim_size(0) == 2
    assert process_mesh.dim_size(-1) == 3
    assert process_mesh.dim_size("x") == 2
    assert process_mesh.dim_size("y") == 3
    assert process_mesh.empty() is False
    assert process_mesh.contains(0) is True
    assert process_mesh.contains(6) is False
    assert process_mesh == process_mesh
    assert process_mesh != process_mesh2
    assert str(process_mesh) == str(process_mesh)

    print("test_auto_process_mesh ... ok")


@run_priority(level="P0")
def test_compute_compatible_process_mesh():
    """test_compute_compatible_process_mesh"""
    process_mesh1 = ProcessMesh([[0, 1, 2], [3, 4, 5]], dim_names=["x", "y"])
    compatible_process_mesh = compute_compatible_process_mesh([process_mesh1, None])
    assert compatible_process_mesh == process_mesh1
    compatible_process_mesh = compute_compatible_process_mesh([None, process_mesh1])
    assert compatible_process_mesh == process_mesh1

    process_mesh2 = ProcessMesh([[0, 1, 2], [3, 4, 5]])
    compatible_process_mesh = compute_compatible_process_mesh([process_mesh1, process_mesh2])
    assert compatible_process_mesh == process_mesh1
    assert compatible_process_mesh == process_mesh2

    process_mesh2 = ProcessMesh([[0, 1, 2, 3, 4, 5]])
    compatible_process_mesh = compute_compatible_process_mesh([process_mesh1, process_mesh2])
    assert compatible_process_mesh == process_mesh1

    process_mesh2 = ProcessMesh([[0, 1, 2]])
    compatible_process_mesh = compute_compatible_process_mesh([process_mesh1, process_mesh2])
    assert compatible_process_mesh == process_mesh1

    print("test_compute_compatible_process_mesh ... ok")


@run_priority(level="P0")
def test_merge_process_mesh():
    """test_merge_process_mesh"""
    process_mesh1 = ProcessMesh([[0, 1, 2], [3, 4, 5]], dim_names=["x", "y"])
    merged_process_mesh = merge_process_mesh([process_mesh1, None])
    assert merged_process_mesh == ProcessMesh([0, 1, 2, 3, 4, 5])

    merged_process_mesh = merge_process_mesh([None, process_mesh1])
    assert merged_process_mesh == ProcessMesh([0, 1, 2, 3, 4, 5])

    process_mesh2 = ProcessMesh([[0, 1, 2], [3, 4, 5]])
    merged_process_mesh = merge_process_mesh([process_mesh1, process_mesh2])
    assert merged_process_mesh == ProcessMesh([0, 1, 2, 3, 4, 5])

    process_mesh2 = ProcessMesh([[0, 1, 2]])
    merged_process_mesh = merge_process_mesh([process_mesh1, process_mesh2])
    assert merged_process_mesh == ProcessMesh([0, 1, 2, 3, 4, 5])

    process_mesh2 = ProcessMesh([[6, 7]])
    merged_process_mesh = merge_process_mesh([process_mesh1, process_mesh2])
    assert merged_process_mesh == ProcessMesh([0, 1, 2, 3, 4, 5, 6, 7])

    print("test_merge_process_mesh ... ok")


if __name__ == "__main__":
    test_auto_process_mesh()
    test_compute_compatible_process_mesh()
    test_merge_process_mesh()
