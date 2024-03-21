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
  * @file dist_ShardingStage3.py
  * @author liujie44@baidu.com
  * @date 2024-02-20
  * @brief
  *
  **************************************************************************/
"""
import paddle
import paddle.distributed as dist
from utils import run_priority

mesh = dist.ProcessMesh([0, 1], dim_names=["x"])


class MLP(paddle.nn.Layer):
    """定义网络结构"""

    def __init__(self):
        """调用父类构造函数"""
        super().__init__()
        self.fc1 = paddle.nn.Linear(8, 8)
        self.fc2 = paddle.nn.Linear(8, 8)

    def forward(self, input):
        """定义前向计算过程"""
        return self.fc2(self.fc1(input))


@run_priority(level="P0")
def test_ShardingStage3():
    """test_ShardingStage3"""

    layer = MLP()
    batch = paddle.rand(shape=[8, 8])
    opt = paddle.optimizer.AdamW(parameters=layer.parameters())
    opt = dist.shard_optimizer(opt, dist.ShardingStage3(mesh))
    for _ in range(5):
        loss = layer(batch)
        loss.backward()
        opt.step()
        opt.clear_grad()
    assert len(loss) == 8
    print("test_ShardingStage3 ... ok")


if __name__ == "__main__":
    test_ShardingStage3()
