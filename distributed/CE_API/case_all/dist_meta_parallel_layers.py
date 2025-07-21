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
  * @file dist_meta_parallel_layers.py
  * @author lvkunpeng@baidu.com
  * @date 2025-07-08
  * @brief
  *
  **************************************************************************/
"""
import pytest
import paddle
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)
from utils import run_priority

class SimpleMPNet(paddle.nn.Layer):
    """Simple MPNet"""
    def __init__(self, vocab_size, hidden_size, inner_size, output_size):
        """__init__"""
        super().__init__()
        self.embedding = VocabParallelEmbedding(vocab_size, hidden_size)
        self.linear1 = ColumnParallelLinear(
            hidden_size,
            inner_size,
            gather_output=False,
            has_bias=True
        )
        self.linear2 = RowParallelLinear(
            inner_size,
            hidden_size,
            input_is_parallel=True,
            has_bias=True
        )
        self.linear3 = paddle.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """forward"""
        x = self.embedding(x)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        return x

@run_priority(level="P0")
def test_meta_parallel_layers():
    """test_meta_parallel_layers"""
    paddle.set_device("gpu")
    fleet.init(is_collective=True)

    vocab_size = 100
    hidden_size = 64
    inner_size = 32
    output_size = 10
    batch_size = 4
    seq_len = 16

    model = SimpleMPNet(vocab_size, hidden_size, inner_size, output_size)
    model = fleet.distributed_model(model) 
    input_data = paddle.randint(0, vocab_size, shape=[batch_size, seq_len])
    output = model(input_data)

    assert isinstance(output, paddle.Tensor)
    assert output.shape == [batch_size, seq_len, output_size]
    print("test_meta_parallel_layers ... ok")


if __name__ == '__main__':
    test_meta_parallel_layers()