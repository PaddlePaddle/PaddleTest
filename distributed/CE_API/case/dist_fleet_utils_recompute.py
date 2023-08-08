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
  * @file dist_fleet_utils_recompute.py
  * @author liujie44@baidu.com
  * @date 2021-11-15 11:15
  * @brief
  *
  **************************************************************************/
"""
import os
import sys
import random
import numpy as np

import paddle
from paddle.distributed.fleet.utils import recompute
from utils import run_priority

# required: gpu


def getFcBlock(block_idx, input_size, is_last=False):
    """getFcBlock"""
    block_name = "block_" + str(block_idx)
    block = paddle.nn.Sequential(
        (block_name + "_fc_0", paddle.nn.Linear(input_size, input_size, bias_attr=False)),
        (block_name + "_dropout", paddle.nn.Dropout(p=0.5)),
        (block_name + "_relu_1", paddle.nn.ReLU()),
        (block_name + "_fc_1", paddle.nn.Linear(input_size, input_size, bias_attr=False)),
        (block_name + "_relu_2", paddle.nn.ReLU()),
    )
    if is_last:
        block.add_sublayer(block_name + "_fc_2", paddle.nn.Linear(input_size, 1, bias_attr=False))
    else:
        block.add_sublayer(block_name + "_fc_2", paddle.nn.Linear(input_size, input_size, bias_attr=False))

    return block


class NaiveFcNet(paddle.nn.Layer):
    """NaiveFcNet"""

    def __init__(self, input_size=10, recompute_blocks=[1, 3], recompute_kwargs={}):
        super(NaiveFcNet, self).__init__()
        self.recompute_blocks = recompute_blocks
        self.recompute_kwargs = recompute_kwargs
        self.runfunc0 = getFcBlock(0, input_size, is_last=False)
        self.runfunc1 = getFcBlock(1, input_size, is_last=False)
        self.runfunc2 = getFcBlock(2, input_size, is_last=False)
        self.runfunc3 = getFcBlock(3, input_size, is_last=False)
        self.runfunc4 = getFcBlock(4, input_size, is_last=True)
        self.total_func = [self.runfunc0, self.runfunc1, self.runfunc2, self.runfunc3, self.runfunc4]

    def forward(self, inputs):
        """forward"""
        nums = len(self.total_func)
        for i in range(nums):
            if i in self.recompute_blocks:
                inputs = recompute(self.total_func[i], inputs, **{"preserve_rng_state": True})
            else:
                inputs = self.total_func[i](inputs)
        return inputs


def run_model(cuda_state, recompute_block=[], recompute_kwargs={}):
    """run_model"""
    gen = paddle.seed(10)
    gen.manual_seed(10)
    np.random.seed(10)
    random.seed(10)
    if cuda_state:
        paddle.set_cuda_rng_state(cuda_state)

    batch_size, input_size = 1, 10
    model = NaiveFcNet(input_size, recompute_blocks=recompute_block, recompute_kwargs=recompute_kwargs)
    optimizer = paddle.optimizer.SGD(learning_rate=0.01, parameters=model.parameters())
    loss_ = []
    param_ = []
    grad_ = []
    for _ in range(5):
        x_data = np.random.randn(batch_size, input_size).astype(np.float32)
        x = paddle.to_tensor(x_data)
        y_pred = model(x)
        loss = y_pred.mean()
        loss_.append(np.asarray(loss).tolist())
        loss.backward()
        optimizer.step()
        param_.append(np.asarray(model.parameters()[9]).tolist())
        grad_.append(np.asarray(model.parameters()[3]._grad_ivar()).tolist())
        optimizer.clear_grad()

    return loss_, param_, grad_


cuda_state = paddle.get_cuda_rng_state()
# without recompute
loss_ref, param_ref, grad_ref = run_model(cuda_state, recompute_block=[])

loss, param, grad = run_model(cuda_state, recompute_block=[1, 2])
print("normal_loss: {}, recompute_loss: {}".format(loss_ref, loss))
# The result of the recompute_loss should be the same as the normal_loss.
assert loss_ref == loss
