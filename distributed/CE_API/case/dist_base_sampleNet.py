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
  * @file sampleNet.py
  * @author liyang109@baidu.com
  * @date 2021-05-10 14:33
  * @brief
  *
  **************************************************************************/
"""
from __future__ import division
from __future__ import print_function
import random
import numpy as np

import paddle
import paddle.distributed as dist
import paddle.distributed.fleet as fleet
from paddle.io import DataLoader, Dataset


def set_random_seed(seed, dp_id, rank_id):
    """Set random seed for reproducability."""
    random.seed(seed)
    np.random.seed(seed + dp_id)
    paddle.seed(seed + rank_id)


vocab_size = 6
hidden_size = 10
inner_size = 8
output_size = 2
seq_length = 2


class SimpleMPNet(paddle.nn.Layer):
    """SimpleMPNet struct"""

    def __init__(self, vocab_xsize, hidden_size, inner_size, output_size, np_fc1, np_fc2, mp_id):
        """init"""
        super(SimpleMPNet, self).__init__()

        if mp_id == 0:
            init_fc1_data = np_fc1[:, : (inner_size // 2)]
            init_fc2_data = np_fc2[: (inner_size // 2), :]
        else:
            init_fc1_data = np_fc1[:, (inner_size // 2) :]
            init_fc2_data = np_fc2[(inner_size // 2) :, :]

        self.linear1 = fleet.meta_parallel.ColumnParallelLinear(
            hidden_size,
            inner_size,
            weight_attr=paddle.framework.ParamAttr(initializer=paddle.nn.initializer.Assign(init_fc1_data)),
            gather_output=False,
            has_bias=True,
        )

        self.linear2 = fleet.meta_parallel.RowParallelLinear(
            inner_size,
            hidden_size,
            weight_attr=paddle.framework.ParamAttr(initializer=paddle.nn.initializer.Assign(init_fc2_data)),
            input_is_parallel=True,
            has_bias=True,
        )

        self.linear3 = paddle.nn.Linear(
            hidden_size,
            output_size,
            weight_attr=paddle.framework.ParamAttr(initializer=paddle.nn.initializer.Constant(0.0)),
            bias_attr=paddle.framework.ParamAttr(initializer=paddle.nn.initializer.Constant(0.0)),
        )

        self.embedding = fleet.meta_parallel.VocabParallelEmbedding(
            vocab_size, hidden_size, weight_attr=paddle.nn.initializer.Constant(value=0.5)
        )

    def forward(self, x):
        """def forward network"""
        x = self.embedding(x)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        return x


class SimpleDPNet(paddle.nn.Layer):
    """simpleDPNet struct"""

    def __init__(self, vocab_size, hidden_size, inner_size, output_size, np_fc1, np_fc2):
        """init"""

        super(SimpleDPNet, self).__init__()
        self.linear1 = paddle.nn.Linear(
            hidden_size,
            inner_size,
            weight_attr=paddle.framework.ParamAttr(initializer=paddle.nn.initializer.Assign(np_fc1)),
            bias_attr=paddle.framework.ParamAttr(initializer=paddle.nn.initializer.Constant(0.0)),
        )

        self.linear2 = paddle.nn.Linear(
            inner_size,
            hidden_size,
            weight_attr=paddle.framework.ParamAttr(initializer=paddle.nn.initializer.Assign(np_fc2)),
            bias_attr=paddle.framework.ParamAttr(initializer=paddle.nn.initializer.Constant(0.0)),
        )

        self.linear3 = paddle.nn.Linear(
            hidden_size,
            output_size,
            weight_attr=paddle.framework.ParamAttr(initializer=paddle.nn.initializer.Constant(0.0)),
            bias_attr=paddle.framework.ParamAttr(initializer=paddle.nn.initializer.Constant(0.0)),
        )

        self.embedding = paddle.nn.Embedding(
            vocab_size, hidden_size, weight_attr=paddle.nn.initializer.Constant(value=0.5)
        )

    def forward(self, x):
        """def forward struct"""
        x = self.embedding(x)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        return x


class TrainDataset(Dataset):
    """train dataset"""

    def __init__(self, length):
        """init"""
        self.length = length

    def __len__(self):
        """len"""
        return self.length

    def __getitem__(self, index):
        """get item"""
        np_input_data = np.random.randint(0, vocab_size, (seq_length,))
        return np_input_data
