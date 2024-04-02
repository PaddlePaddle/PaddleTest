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
  * Copyright (c) 2024 Baidu.com, Inc. All Rights Reserved
  * @file dist_Strategy.py
  * @author liujie44@baidu.com
  * @date 2024-02-21
  * @brief
  *
  **************************************************************************/
"""
import paddle
import paddle.distributed as dist
from utils import run_priority


@run_priority(level="P0")
def test_Strategy():
    """test_Strategy"""

    strategy = dist.Strategy()

    strategy.sharding.enable = True
    strategy.sharding.stage = 2
    strategy.sharding.degree = 2

    strategy.gradient_merge.enable = True
    strategy.gradient_merge.k_steps = 2
    strategy.gradient_merge.avg = False

    strategy.pipeline.enable = True
    strategy.pipeline.schedule_mode = "1F1B"  # default is "1F1B"
    strategy.pipeline.micro_batch_size = 2

    print("test_Strategy ... ok")


def test_Strategy_sharding():
    """test_Strategy_sharding"""
    strategy = dist.Strategy()

    strategy.sharding.enable = True
    strategy.sharding.stage = 2
    strategy.sharding.degree = 2

    print("test_Strategy_sharding ... ok")


def test_Strategy_fused_passes():
    """test_Strategy_fused_passes"""
    strategy = dist.Strategy()

    strategy.fused_passes.enable = True
    strategy.fused_passes.gemm_spilogue = True
    strategy.fused_passes.dropout_add = True

    print("test_Strategy_fused_passes ... ok")


def test_Strategy_gradient_merge():
    """test_Strategy_gradient_merge"""
    strategy = dist.Strategy()

    strategy.gradient_merge.enable = True
    strategy.gradient_merge.k_steps = 2
    strategy.gradient_merge.avg = True

    print("test_Strategy_gradient_merge ... ok")


def test_Strategy_pipeline():
    """test_Strategy_pipeline"""
    strategy = dist.Strategy()

    strategy.pipeline.enable = True
    strategy.pipeline.micro_batch_size = 2

    print("test_Strategy_pipeline ... ok")


if __name__ == "__main__":
    test_Strategy()
    test_Strategy_sharding()
    test_Strategy_fused_passes()
    test_Strategy_gradient_merge()
    test_Strategy_pipeline()
