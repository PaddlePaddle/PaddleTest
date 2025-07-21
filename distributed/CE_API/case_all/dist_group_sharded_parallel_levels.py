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
  * @file dist_group_sharded_parallel_levels.py
  * @author lvkunpeng@baidu.com
  * @date 2025-07-04
  * @brief
  *
  **************************************************************************/
"""
import pytest
import paddle
from paddle.nn import Linear
from paddle.distributed import fleet
from paddle.distributed.sharding import group_sharded_parallel

from paddle.distributed.fleet.meta_parallel.sharding.group_sharded_stage2 import GroupShardedStage2
from paddle.distributed.fleet.meta_parallel.sharding.group_sharded_optimizer_stage2 import GroupShardedOptimizerStage2
from paddle.distributed.fleet.meta_parallel.sharding.group_sharded_stage3 import GroupShardedStage3

@pytest.mark.parametrize("level", ["os", "os_g", "p_g_os"])
def test_group_sharded_parallel_levels(level):
    """test_group_sharded_parallel_levels"""
    fleet.init(is_collective=True)
    group = paddle.distributed.new_group([0, 1])
    model = Linear(1000, 1000)

    clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=1.0)
    optimizer = paddle.optimizer.AdamW(
        learning_rate=0.001,
        parameters=model.parameters(),
        weight_decay=0.00001,
        grad_clip=clip
    )
    scaler = None
    model, optimizer, scaler = group_sharded_parallel(model, optimizer, level, group=group, scaler=scaler)

    img = paddle.randn([16, 1000])
    label = paddle.randint(0, 1000, shape=[16], dtype='int64')
    img.stop_gradient = True
    label.stop_gradient = True

    out = model(img)
    loss = paddle.nn.functional.cross_entropy(input=out, label=label)

    loss.backward()
    optimizer.step()
    optimizer.clear_grad()

    assert isinstance(loss, paddle.Tensor)
    assert loss.numpy().ndim == 0

    if level in ["os", "os_g"]:
        assert isinstance(optimizer, GroupShardedOptimizerStage2)
        assert isinstance(model, GroupShardedStage2)
    elif level == "p_g_os":
        assert isinstance(model, GroupShardedStage3)


if __name__ == "__main__":
    test_group_sharded_parallel_levels("os")
    test_group_sharded_parallel_levels("os_g")
    test_group_sharded_parallel_levels("p_g_os")

