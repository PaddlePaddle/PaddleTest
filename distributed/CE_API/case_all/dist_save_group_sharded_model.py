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
  * @file dist_save_group_sharded_model.py
  * @author lvkunpeng@baidu.com
  * @date 2025-07-04
  * @brief
  *
  **************************************************************************/
"""
import os
import shutil
import tempfile

import paddle
from paddle.nn import Linear
from paddle.distributed import fleet
from paddle.distributed.sharding import group_sharded_parallel, save_group_sharded_model
from utils import run_priority

@run_priority(level="P0")
def test_save_group_sharded_model():
    """test_save_group_sharded_model"""
    fleet.init(is_collective=True)
    group = paddle.distributed.new_group([0, 1])
    model = Linear(1000, 1000)

    clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=1.0)
    optimizer = paddle.optimizer.AdamW(
        learning_rate=0.001,
        parameters=model.parameters(),
        weight_decay=0.00001,
        grad_clip=clip
    )

    scaler = None 
    model, optimizer, scaler = group_sharded_parallel(model, optimizer, "p_g_os", group=group, scaler=scaler)

    img = paddle.randn([16, 1000])
    label = paddle.randint(0, 1000, shape=[16], dtype='int64')
    img.stop_gradient = True
    label.stop_gradient = True

    out = model(img)
    loss = paddle.nn.functional.cross_entropy(input=out, label=label)

    loss.backward()
    optimizer.step()
    optimizer.clear_grad()

    output_dir = tempfile.mkdtemp()
    save_group_sharded_model(model, output_dir, optimizer=optimizer)
    saved_files = os.listdir(output_dir)
    assert len(saved_files) > 0

    shutil.rmtree(output_dir)


if __name__ == '__main__':
    test_save_group_sharded_model()