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
  * Copyright (c) 2024 Baidu.com, Inc. All Rights Reserved
  * @file dist_shard_layer.py
  * @author liujie44@baidu.com
  * @date 2024-02-21
  * @brief
  *
  **************************************************************************/
"""
import paddle
import paddle.distributed as dist
from utils import run_priority

mesh = dist.ProcessMesh([0, 1], dim_names=["x"])


class MLP(paddle.nn.Layer):
    """MLP"""

    def __init__(self):
        """__init__"""
        super().__init__()
        self.fc1 = paddle.nn.Linear(8, 8)
        self.fc2 = paddle.nn.Linear(8, 8)

    def forward(self, input):
        """forward"""
        return self.fc2(self.fc1(input))


def shard_fn(layer_name, layer, process_mesh):
    """shard_fn"""
    if layer_name == "fc1":
        layer.weight = dist.shard_tensor(layer.weight, process_mesh, [dist.Shard(0)])


@run_priority(level="P0")
def test_shard_layer():
    """test_shard_layer"""
    layer = MLP()
    layer = dist.shard_layer(layer, mesh, shard_fn)
    print(layer)

    print("test_shard_layer ... ok")


if __name__ == "__main__":
    test_shard_layer()
