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
  * @file dist_enable_auto_dp.py
  * @author lvkunpeng@baidu.com
  * @date 2025-07-08
  * @brief
  *
  **************************************************************************/
"""
import io
import contextlib
import numpy as np
import paddle
from paddle import nn
import paddle.distributed as dist
from paddle.io import Dataset, DataLoader
from paddle.distributed import enable_auto_dp

from utils import run_priority

enable_auto_dp()

BATCH_SIZE = 32
CLASS_NUM = 10
INPUT_DIM = 256
STEPS = 100

class RandomDataset(Dataset):  # type: ignore[type-arg]
    def __init__(self, num_samples):
        rank = dist.get_rank() if dist.get_world_size() > 1 else 0
        np.random.seed(42 + rank)
        self.num_samples = num_samples
    def __getitem__(self, idx):
        x = np.random.rand(INPUT_DIM).astype('float32')
        y = np.random.randint(0, CLASS_NUM, (1,)).astype('int64')
        return x, y
    def __len__(self):
        return self.num_samples

class SimpleNet(nn.Layer):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, 102400),
            nn.Linear(102400, INPUT_DIM),
            nn.Linear(INPUT_DIM, CLASS_NUM),
        )
    def forward(self, x):
        return self.net(x)

@run_priority(level="P0")
def test_enable_auto_dp():
    """test_enable_auto_dp"""

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        model = SimpleNet()
        optimizer = paddle.optimizer.AdamW(learning_rate=1e-3, parameters=model.parameters())
        dataset = RandomDataset(num_samples=STEPS * BATCH_SIZE)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

        model.train()
        for step, (x, y) in enumerate(loader):
            y.stop_gradient = True
            loss = paddle.mean(model(x))
            loss.backward()
            optimizer.step()
            model.clear_gradients()
            if step % 5 == 0:
                print(f"[step {step}] loss: {loss.item():.4f}")

    captured = buffer.getvalue()
    assert "loss" in captured
    print("test enable auto dp ... ok")

if __name__ == '__main__':
    test_enable_auto_dp()