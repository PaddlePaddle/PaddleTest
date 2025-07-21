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
  * @file dist_fleet_step.py
  * @author lvkunpeng@baidu.com
  * @date 2025-07-04
  * @brief
  *
  **************************************************************************/
"""
import paddle
import paddle.nn as nn
from paddle.distributed import fleet
from utils import run_priority

class LinearNet(nn.Layer):
    """LinearNet"""

    def __init__(self):
        super().__init__()
        self._linear1 = nn.Linear(10, 10)
        self._linear2 = nn.Linear(10, 1)

    def forward(self, x):
        """forward"""
        return self._linear2(self._linear1(x))

@run_priority(level="P0")
def test_fleet_step():
    """test_fleet_step"""
    # 1. initialize fleet environment
    fleet.init(is_collective=True)

    # 2. create model & optimizer
    layer = LinearNet()
    loss_fn = nn.MSELoss()
    adam = paddle.optimizer.Adam(learning_rate=0.001, parameters=layer.parameters())

    # 3. wrap for distributed
    dp_layer = fleet.distributed_model(layer)
    adam = fleet.distributed_optimizer(adam)

    # 4. run layer
    inputs = paddle.randn([10, 10], dtype='float32')
    labels = paddle.randn([10, 1], dtype='float32')

    outputs = dp_layer(inputs)
    loss = loss_fn(outputs, labels)
    print("loss:", loss.numpy())
    loss.backward()

    params_before = [p.clone() for p in dp_layer.parameters()]

    adam.step()
    adam.clear_grad()

    updated = False
    for p_before, p_after in zip(params_before, dp_layer.parameters()):
        if not paddle.allclose(p_before, p_after):
            updated = True
            break

    assert updated

if __name__ == "__main__":
    test_fleet_step()