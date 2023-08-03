#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
/***************************************************************************
  *
  * Copyright (c) 2022 Baidu.com, Inc. All Rights Reserved
  * @file dist_fleet_distributed_model.py
  * @author liujie44@baidu.com
  * @date 2022-04-15 11:10
  * @brief
  * disable
  **************************************************************************/
"""
import numpy as np
import paddle
import paddle.distributed as dist
import paddle.distributed.fleet as fleet

from utils import run_priority


@run_priority(level="P0")
def test_fleet_distributed_model():
    """test_fleet_distributed_model"""
    fleet.init(is_collective=True)
    if fleet.worker_index() == 0:
        np.random.seed(2021)
    else:
        np.random.seed(2022)

    net = paddle.nn.Sequential(paddle.nn.Linear(10, 1), paddle.nn.Linear(1, 2))
    net = dist.fleet.distributed_model(net)
    data = np.random.uniform(-1, 1, [30, 10]).astype("float32")
    data = paddle.to_tensor(data)
    loss = net(data)
    print(loss)


if __name__ == "__main__":
    test_fleet_distributed_model()
