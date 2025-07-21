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
  * @file dist_fleet_get_lr.py
  * @author lvkunpeng@baidu.com
  * @date 2025-07-04
  * @brief
  *
  **************************************************************************/
"""
import paddle
import paddle.nn as nn
import numpy as np
from paddle.distributed import fleet
from utils import run_priority


@run_priority(level="P0")
def test_fleet_set_lr():
    """test_fleet_set_lr"""
    fleet.init(is_collective=True)

    value = np.arange(26).reshape(2, 13).astype("float32")
    a = paddle.to_tensor(value)

    layer = paddle.nn.Linear(13, 5)
    adam = paddle.optimizer.Adam(learning_rate=0.01, parameters=layer.parameters())

    adam = fleet.distributed_optimizer(adam)
    dp_layer = fleet.distributed_model(layer)

    lr_list = [0.2, 0.3, 0.4, 0.5, 0.6]

    for expected_lr in lr_list:
        adam.set_lr(expected_lr)
        lr = adam.get_lr()
        assert lr == expected_lr

        if fleet.worker_index() == 0:
            print("test_fleet_set_lr ... ok")


if __name__ == "__main__":
    test_fleet_set_lr()
