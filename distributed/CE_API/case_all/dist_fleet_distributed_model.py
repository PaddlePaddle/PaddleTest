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
  * @file dist_fleet_distributed_model.py
  * @author lvkunpeng@baidu.com
  * @date 2025-07-18
  * @brief
  *
  **************************************************************************/
"""
import paddle
import paddle.nn as nn
import numpy as np
from paddle.distributed import fleet
from utils import run_priority


class LinearNet(nn.Layer):
    def __init__(self):
        super().__init__()
        self._linear1 = nn.Linear(10, 10)
        self._linear2 = nn.Linear(10, 1)

    def forward(self, x):
        return self._linear2(self._linear1(x))

@run_priority(level="P0")
def test_fleet_distributed_model():
    """test_fleet_distributed_model"""
    # 1. initialize fleet environment
    fleet.init(is_collective=True)

    # 2. create layer & optimizer
    layer = LinearNet()
    loss_fn = nn.MSELoss()
    adam = paddle.optimizer.Adam(
        learning_rate=0.001, parameters=layer.parameters())

    # 3. get data_parallel model using fleet
    adam = fleet.distributed_optimizer(adam)
    dp_layer = fleet.distributed_model(layer)

    # 4. run layer
    inputs = paddle.randn([10, 10], 'float32')
    outputs = dp_layer(inputs)

    assert outputs.shape == [10, 1]

    labels = paddle.randn([10, 1], 'float32')
    loss = loss_fn(outputs, labels)

    print("loss:", loss.numpy())

    loss.backward()
    adam.step()
    adam.clear_grad()

    loss_value = loss.numpy()
    assert isinstance(loss_value, np.ndarray)
    assert loss_value.size == 1
    assert loss_value >= 0
    print("test_fleet_distributed_model ... ok")


if __name__ == '__main__':
    test_fleet_distributed_model()