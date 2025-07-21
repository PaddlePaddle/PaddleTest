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
  * @file dist_local_map.py
  * @author lvkunpeng@baidu.com
  * @date 2025-07-10
  * @brief
  *
  **************************************************************************/
"""
from __future__ import annotations
import paddle
import paddle.distributed as dist
from paddle import Tensor
from paddle.distributed import ProcessMesh, local_map
from utils import run_priority


def custom_function(x):
    """custom_function"""
    mask = paddle.zeros_like(x)
    if dist.get_rank() == 0:
        mask[1:3] = 1
    else:
        mask[4:7] = 1
    x = x * mask
    mask_sum = paddle.sum(x)
    mask_sum = mask_sum / mask.sum()
    return mask_sum

@run_priority(level="P0")
def test_local_map():
    """test_local_map"""
    dist.init_parallel_env()
    mesh = ProcessMesh([0, 1], dim_names=["x"])
    local_input = paddle.arange(0, 10, dtype="float32")
    local_input = local_input + dist.get_rank()
    input_dist = dist.auto_parallel.api.dtensor_from_local(
        local_input, mesh, [dist.Shard(0)]
    )
    wrapped_func = dist.local_map(
        custom_function,
        out_placements=[[dist.Partial(dist.ReduceType.kRedSum)]],
        in_placements=[[dist.Shard(0)]],
        process_mesh=mesh
    )

    output_dist = wrapped_func(input_dist)
    local_value = output_dist._local_value()
    gathered_values: list[Tensor] = []
    dist.all_gather(gathered_values, local_value)

    if dist.get_rank() == 0:
        assert gathered_values[0] == 1.5
    if dist.get_rank() == 1:
        assert gathered_values[1] == 6.0
    assert output_dist == 7.5
    print("test_local_map ... ok")


if __name__ == "__main__":
    test_local_map()