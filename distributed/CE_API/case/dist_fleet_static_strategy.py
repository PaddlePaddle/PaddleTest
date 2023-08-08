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
  * @file test_dist_fleet_static_strategy.py
  * @author liyang109@baidu.com
  * @date 2020-11-16 14:41
  * @brief
  *
  **************************************************************************/
"""
import paddle
import paddle.distributed.fleet as fleet

paddle.enable_static()


def test_dist_strategy():
    """dist strategy"""
    strategy = fleet.DistributedStrategy()
    strategy.amp = True
    strategy.amp_configs = {
        "init_loss_scaling": 10240,
        "decr_every_n_nan_or_inf": 2,
        "incr_every_n_steps": 1000,
        "incr_ratio": 3.0,
        "use_dynamic_loss_scaling": True,
        "decr_ratio": 0.5,
    }
    strategy.recompute = True
    strategy.recompute_configs = {"checkpoints": ["a", "b", "c"]}
    strategy.pipeline = True
    strategy.pipeline_configs = {"accumulate_steps": 10}
    strategy.localsgd = True
    strategy.localsgd_configs = {"k_steps": 10}
    strategy.dgc = True
    strategy.sync_nccl_allreduce = True
    strategy.nccl_comm_num = 2
    strategy.use_hierarchical_allreduce = True
    strategy.hierarchical_allreduce_inter_nranks = 8
    strategy.sync_batch_norm = True
    strategy.fuse_all_reduce_ops = True
    strategy.fuse_grad_size_in_MB = 50
    strategy._fuse_grad_size_in_TFLOPS = 0.1
    strategy.gradient_merge = True
    strategy.gradient_merge_configs = {"k_steps": 10}
    strategy.lars = True
    strategy.lamb = True
    strategy.a_sync = False
    strategy.a_sync_configs = {"k_steps": 1}
    strategy.auto = True

    build_strategy = paddle.static.BuildStrategy()
    build_strategy.enable_sequential_execution = True
    build_strategy.fuse_elewise_add_act_ops = True
    build_strategy.fuse_bn_act_ops = True
    build_strategy.enable_auto_fusion = True
    build_strategy.fuse_relu_depthwise_conv = True
    build_strategy.fuse_broadcast_ops = True
    build_strategy.fuse_all_optimizer_ops = True
    build_strategy.sync_batch_norm = True
    build_strategy.enable_inplace = True

    exe_strategy = paddle.static.ExecutionStrategy()
    exe_strategy.num_threads = 10
    exe_strategy.num_iteration_per_drop_scope = 10
    exe_strategy.num_iteration_per_run = 10

    strategy.execution_strategy = exe_strategy
    strategy.build_strategy = build_strategy

    print(strategy.lars)
    strategy.save_to_prototxt("origin_dist_strategy.prototxt")

    new_strategy = fleet.DistributedStrategy()
    new_strategy.load_from_prototxt("origin_dist_strategy.prototxt")
    print(new_strategy.lars)
    assert strategy.lars == new_strategy.lars
    assert strategy.recompute == new_strategy.recompute
    assert strategy.lamb == new_strategy.lamb
    assert strategy.a_sync == new_strategy.a_sync
    assert strategy.auto == new_strategy.auto
    assert strategy.gradient_merge == new_strategy.gradient_merge
    assert strategy.nccl_comm_num == new_strategy.nccl_comm_num
    assert strategy.sync_batch_norm == new_strategy.sync_batch_norm
