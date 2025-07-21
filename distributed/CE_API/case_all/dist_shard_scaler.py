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
  * @file dist_shard_scaler.py
  * @author lvkunpeng@baidu.com
  * @date 2025-07-04
  * @brief
  *
  **************************************************************************/
"""
import paddle
import paddle.distributed as dist
from paddle.distributed import shard_scaler
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


@run_priority(level="P0")
def test_shard_scaler():
    """test_shard_scaler"""
    layer = MLP()
    layer = dist.shard_layer(layer, mesh)
    batch = paddle.rand(shape=[8, 8])
    opt = paddle.optimizer.AdamW(parameters=layer.parameters())
    layer, opt = paddle.amp.decorate(layer, opt, level='O2')
    scaler = paddle.amp.GradScaler(init_loss_scaling=1024)
    scaler = shard_scaler(scaler)
    opt = dist.shard_optimizer(opt)
    for _ in range(5):
        with paddle.amp.auto_cast(True):
            loss = layer(batch)
        scaled = scaler.scale(loss)
        scaled.backward()
        scaler.step(opt)
        scaler.update()
        opt.clear_grad()
        assert loss.shape == [8, 8]
        assert scaled.shape == loss.shape

    print("test_shard_scaler ... ok")


if __name__ == "__main__":
    test_shard_scaler()
