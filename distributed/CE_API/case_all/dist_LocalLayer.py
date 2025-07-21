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
  * @file dist_LocalLayer.py
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
from paddle.distributed import ProcessMesh, LocalLayer
from utils import run_priority


class CustomLayer(dist.LocalLayer):
    """"CustomLayer"""
    def __init__(self, out_dist_attrs, grad_dist_attrs):
        """__init__"""
        super().__init__(out_dist_attrs, grad_dist_attrs)
        self.local_result = paddle.to_tensor(0.0)

    def forward(self, x):
        """forward"""
        mask = paddle.zeros_like(x)
        if dist.get_rank() == 0:
            mask[1:3] = 1
        else:
            mask[4:7] = 1
        x = x * mask
        mask_sum = paddle.sum(x)
        mask_sum = mask_sum / mask.sum()
        self.local_result = mask_sum
        return mask_sum

@run_priority(level="P0")
def test_LocalLayer():
    """test_LocalLayer"""
    dist.init_parallel_env()
    mesh = ProcessMesh([0, 1], dim_names=["x"])
    dist_attrs = [
        (mesh, [dist.Partial(dist.ReduceType.kRedSum)]),
    ]
    local_input = paddle.arange(0, 10, dtype="float32")
    local_input = local_input + dist.get_rank()
    input_dist = dist.auto_parallel.api.dtensor_from_local(
        local_input, mesh, [dist.Shard(0)]
    )
    custom_layer = CustomLayer(dist_attrs, dist_attrs)
    output_dist = custom_layer(input_dist)

    local_value = custom_layer.local_result
    gathered_values: list[Tensor] = []
    dist.all_gather(gathered_values, local_value)

    if dist.get_rank() == 0:
        assert gathered_values[0] == 1.5
    if dist.get_rank() == 1:
        assert gathered_values[1] == 6.0
    assert output_dist == 7.5
    print("test_LocalLayer ... ok")


if __name__ == "__main__":
    test_LocalLayer()