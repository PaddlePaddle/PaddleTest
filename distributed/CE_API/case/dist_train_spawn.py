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
  * @file dist_train_spawn.py
  * @author liujie44@baidu.com
  * @date 2021-11-09 11:10
  * @brief
  *
  **************************************************************************/
"""
from __future__ import print_function
import os
import paddle
import paddle.nn as nn
import paddle.optimizer as opt
import paddle.distributed as dist


class LinearNet(nn.Layer):
    """linear net"""

    def __init__(self):
        super(LinearNet, self).__init__()
        self._linear1 = nn.Linear(10, 10)
        self._linear2 = nn.Linear(10, 1)

    def forward(self, x):
        """forward"""
        return self._linear2(self._linear1(x))


def train(print_result=True):
    """train"""
    # 1. initialize parallel environment
    dist.init_parallel_env()

    # 2. create data parallel layer & optimizer
    layer = LinearNet()
    dp_layer = paddle.DataParallel(layer)

    loss_fn = nn.MSELoss()
    adam = opt.Adam(learning_rate=0.001, parameters=dp_layer.parameters())

    # 3. run layer
    inputs = paddle.randn([10, 10], "float32")
    outputs = dp_layer(inputs)
    labels = paddle.randn([10, 1], "float32")
    loss = loss_fn(outputs, labels)

    if print_result is True:
        print("loss:", loss.numpy())
    assert loss.numpy() != 0

    loss.backward()

    adam.step()
    adam.clear_grad()
    print("test_train_spawn ... ok")


if __name__ == "__main__":
    dist.spawn(train)
    dist.spawn(train, args=(True,))
    dist.spawn(train, args=(True,), nprocs=2)
    dist.spawn(train, args=(True,), nprocs=2, gpus="0,1")
