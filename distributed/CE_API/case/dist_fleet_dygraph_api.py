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
  * @file dist_fleet_dygraph_api.py.py
  * @author liyang109@baidu.com
  * @date 2020-11-16 14:38
  * @brief
  *
  **************************************************************************/
"""
import paddle
import paddle.nn as nn
from paddle.distributed import fleet


class LinearNet(nn.Layer):
    """linera net"""

    def __init__(self):
        super(LinearNet, self).__init__()
        self._linear1 = nn.Linear(10, 10)
        self._linear2 = nn.Linear(10, 1)

    def forward(self, x):
        """forward"""
        return self._linear2(self._linear1(x))


# 1. enable dynamic mode
# paddle.disable_static()

# 2. initialize fleet environment
fleet.init(is_collective=True)

# 3. create layer & optimizer
layer = paddle.nn.Linear(10, 10)
adam = paddle.optimizer.Adam(learning_rate=0.001, parameters=layer.parameters())

# 4. get data_parallel model using fleet
adam = fleet.distributed_optimizer(adam)
dp_layer = fleet.distributed_model(layer)

for step in range(1):
    inputs = paddle.randn([10, 10], "float32")
    outputs = dp_layer(inputs)
    loss = paddle.mean(outputs)

    print("step:{}\tloss:{}".format(step, loss.numpy()))

    loss.backward()

    adam.step()
    adam.clear_grad()
    state_dict = adam.state_dict()
    paddle.framework.save(state_dict, "paddle_dy")
    # opti_state_dict = paddle.framework.load("paddle_dy")
    # adam.set_state_dict(opti_state_dict)
