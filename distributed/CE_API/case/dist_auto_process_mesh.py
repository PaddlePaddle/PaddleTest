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
  * @file dist_auto_process_mesh.py
  * @author liujie44@baidu.com
  * @date 2021-11-09 11:00
  * @brief
  *
  **************************************************************************/
"""
import sys
import numpy as np
import paddle
import paddle.nn.functional as F
from paddle import nn, static
from paddle.distributed.auto_parallel.process_mesh import (
    ProcessMesh,
    compute_compatible_process_mesh,
    merge_process_meshes,
)
from paddle.distributed.auto_parallel.static.dist_context import (
    get_default_distributed_context,
)

from utils import run_priority

paddle.enable_static()

batch_size = 4
epoch_num = 10
hidden_size = 1024
sequence_len = 512


class MLPLayer(nn.Layer):
    """MLPLayer"""

    def __init__(
        self,
        hidden_size=1024,
        intermediate_size=4 * 1024,
        dropout_ratio=0.1,
        initializer_range=0.02,
    ):
        super().__init__()
        d_model = hidden_size
        dim_feedforward = intermediate_size
        param_initializer = nn.initializer.Normal(mean=0.0, std=initializer_range)

        self.norm = nn.LayerNorm(d_model, epsilon=1e-5)
        self.linear0 = nn.Linear(
            d_model,
            dim_feedforward,
            weight_attr=paddle.ParamAttr(initializer=param_initializer),
            bias_attr=None,
        )
        self.linear1 = nn.Linear(
            dim_feedforward,
            d_model,
            weight_attr=paddle.ParamAttr(initializer=param_initializer),
            bias_attr=None,
        )

    def forward(self, input):
        """forward"""
        out = self.norm(input)
        out = self.linear0(out)
        out = F.gelu(out, approximate=True)
        out = self.linear1(out)
        return out


@run_priority(level="P0")
def test_auto_process_mesh_construction():
    """test_auto_process_mesh_construction"""
    mesh = [[0, 1, 2], [3, 4, 5]]
    process_mesh = ProcessMesh(mesh, dim_names=["x", "y"])
    assert process_mesh.shape == [2, 3]
    assert process_mesh.process_ids == [0, 1, 2, 3, 4, 5]
    assert process_mesh.dim_names == ["x", "y"]
    assert process_mesh.ndim == 2
    assert process_mesh == process_mesh
    assert str(process_mesh) == str(process_mesh)

    sub_process_mesh1 = process_mesh[0]
    assert sub_process_mesh1.shape == [3]
    assert sub_process_mesh1.process_ids == [0, 1, 2]
    assert sub_process_mesh1.dim_names == ["y"]
    assert sub_process_mesh1.ndim == 1

    sub_process_mesh2 = process_mesh[:, 1]
    assert sub_process_mesh2.shape == [2]
    assert sub_process_mesh2.process_ids == [1, 4]
    assert sub_process_mesh2.dim_names == ["x"]
    assert sub_process_mesh2.ndim == 1

    sub_process_mesh3 = sub_process_mesh2[:]
    assert sub_process_mesh3.shape == [2]
    assert sub_process_mesh3.process_ids == [1, 4]
    assert sub_process_mesh3.dim_names == ["x"]
    assert sub_process_mesh3.ndim == 1

    sub_process_mesh4 = process_mesh[1, 1]
    assert sub_process_mesh4.shape == [1]
    assert sub_process_mesh4.process_ids == [4]
    assert sub_process_mesh4.dim_names == ["d0"]
    assert sub_process_mesh4.ndim == 1

    sub_process_mesh5 = sub_process_mesh3[0]
    assert sub_process_mesh5.shape == [1]
    assert sub_process_mesh5.process_ids == [1]
    assert sub_process_mesh5.dim_names == ["d0"]
    assert sub_process_mesh5.ndim == 1

    print("test_auto_process_mesh_construction ... ok")


@run_priority(level="P0")
def test_auto_process_mesh_context_manager():
    """test_auto_process_mesh_context_manager"""
    mesh = np.array([1, 2, 3, 4])
    input = static.data(
        name="input",
        shape=[batch_size, sequence_len, hidden_size],
        dtype="float32",
    )
    # label = static.data(name="label", shape=[batch_size, sequence_len, 1], dtype="float32")

    mlp = MLPLayer(
        hidden_size=hidden_size,
        intermediate_size=4 * hidden_size,
        dropout_ratio=0.1,
        initializer_range=0.02,
    )

    with ProcessMesh(mesh, ["d"]):
        out = mlp(input)
        print(out)

    default_program = paddle.static.default_main_program()
    default_dist_context = get_default_distributed_context()

    for block in default_program.blocks:
        for tensor in block.vars.values():
            dist_tensor = default_dist_context.get_dist_tensor_for_program(tensor)
            if dist_tensor is not None:
                assert dist_tensor.dist_attr.process_mesh == ProcessMesh(mesh)
        for op in block.ops:
            dist_op = default_dist_context.get_dist_op_for_program(op)
            if dist_op is not None:
                assert dist_op.dist_attr.process_mesh == ProcessMesh(mesh)

    print("test_auto_process_mesh_context_manager ... ok")


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
def test_merge_process_meshes():
    """test_merge_process_meshes"""
    process_mesh1 = ProcessMesh([[0, 1, 2], [3, 4, 5]], dim_names=["x", "y"])
    merged_process_mesh = merge_process_meshes([process_mesh1, None])
    assert merged_process_mesh == ProcessMesh([0, 1, 2, 3, 4, 5])

    merged_process_mesh = merge_process_meshes([None, process_mesh1])
    assert merged_process_mesh == ProcessMesh([0, 1, 2, 3, 4, 5])

    process_mesh2 = ProcessMesh([[0, 1, 2], [3, 4, 5]])
    merged_process_mesh = merge_process_meshes([process_mesh1, process_mesh2])
    assert merged_process_mesh == ProcessMesh([0, 1, 2, 3, 4, 5])

    process_mesh2 = ProcessMesh([[0, 1, 2]])
    merged_process_mesh = merge_process_meshes([process_mesh1, process_mesh2])
    assert merged_process_mesh == ProcessMesh([0, 1, 2, 3, 4, 5])

    process_mesh2 = ProcessMesh([[6, 7]])
    merged_process_mesh = merge_process_meshes([process_mesh1, process_mesh2])
    assert merged_process_mesh == ProcessMesh([0, 1, 2, 3, 4, 5, 6, 7])

    print("test_merge_process_meshes ... ok")


if __name__ == "__main__":
    test_auto_process_mesh_construction()
    test_auto_process_mesh_context_manager()
    test_compute_compatible_process_mesh()
    test_merge_process_meshes()
