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
  * @file dist_create_mesh.py
  * @author lvkunpeng@baidu.com
  * @date 2025-07-09
  * @brief
  *
  **************************************************************************/
"""
import numpy as np
import paddle
from functools import reduce
import pytest
from paddle.distributed import ProcessMesh
from paddle.distributed.auto_parallel import create_mesh

from utils import run_priority


@run_priority(level="P0")
def test_create_mesh_basic():
    """test create mesh with 2 dimensions"""
    mesh_dims = [("x", 2), ("y", 3)]

    mesh = create_mesh(mesh_dims)
    assert isinstance(mesh, ProcessMesh)

    print("mesh.shape:", mesh.shape)
    print("mesh.dim_names:", mesh.dim_names)

    expected_arr = [0, 1, 2, 3, 4, 5]
    np.testing.assert_array_equal(mesh.process_ids, expected_arr)
    print("test_create_mesh_basic ... ok")

@run_priority(level="P0")
def test_create_mesh_single_dim():
    """test create mesh with single dimension"""
    mesh_dims = [("z", 4)]
    mesh = create_mesh(mesh_dims)
    assert isinstance(mesh, ProcessMesh)
    assert mesh.dim_names == ["z"]
    assert mesh.shape == [4]
    np.testing.assert_array_equal(mesh.process_ids, np.arange(4))
    print("test_create_mesh_single_dim ... ok")

@run_priority(level="P0")
def test_create_mesh_empty_dims():
    """test create mesh with empty dims"""
    mesh_dims = []
    with pytest.raises(ValueError):
        create_mesh(mesh_dims)
    print("test_create_mesh_empty_dims ... ok")


if __name__ == '__main__':
    test_create_mesh_basic()
    test_create_mesh_single_dim()
    test_create_mesh_empty_dims()