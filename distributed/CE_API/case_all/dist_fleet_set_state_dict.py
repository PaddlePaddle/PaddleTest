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
import os
import numpy as np
import paddle
from paddle.distributed import fleet
from utils import run_priority

fleet.init(is_collective=True)

@run_priority(level="P0")
def test_fleet_set_state_dict():
    """test_fleet_set_state_dict"""
    value = np.arange(26).reshape(2, 13).astype("float32")
    a = paddle.to_tensor(value)

    layer = paddle.nn.Linear(13, 5)
    adam = paddle.optimizer.Adam(learning_rate=0.01, parameters=layer.parameters())

    adam = fleet.distributed_optimizer(adam)
    dp_layer = fleet.distributed_model(layer)

    out = dp_layer(a)
    loss = out.mean()
    loss.backward()
    adam.step()
    adam.clear_grad()

    state_dict = adam.state_dict()
    paddle.save(state_dict, "paddle_dy")
    assert os.path.exists("paddle_dy")

    para_state_dict = paddle.load( "paddle_dy")
    adam.set_state_dict(para_state_dict)

    new_state_dict = adam.state_dict()
    for k in state_dict:
        assert k in new_state_dict
        np.testing.assert_allclose(state_dict[k], new_state_dict[k])
    print("test_fleet_set_state_dict ... ok")


if __name__ == '__main__':
    test_fleet_set_state_dict()