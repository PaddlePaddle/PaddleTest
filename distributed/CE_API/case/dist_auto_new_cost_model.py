#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
# ======================================================================
#
# Copyright (c) 2022 Baidu.com, Inc. All Rights Reserved
#
# ======================================================================
"""
/***************************************************************************
  *
  * Copyright (c) 2022 Baidu.com, Inc. All Rights Reserved
  * @file dist_auto_new_cost_model.py
  * @author liujie44@baidu.com
  * @date 2022-09-07 11:00
  * @brief
  *
  **************************************************************************/
"""
import json
import os
import tempfile

import numpy as np
import paddle
import paddle.distributed.auto_parallel.static.cost as cost_model
from paddle.distributed.auto_parallel.static.cluster import Cluster
from paddle.distributed.auto_parallel.static.cost import CommContext
from paddle.distributed.auto_parallel.static.cost.base_cost import (
    build_comp_desc_from_op,
    build_comp_desc_str_for_predict,
    calc_time_by_modeling,
)

from dist_auto_cluster import cluster_json
from utils import run_priority

paddle.enable_static()


def check_cost(cost):
    """check cost"""
    if cost.memory >= 0 and cost.flops >= 0 and cost.time >= 0:
        print("SUCCESS")
        return True
    print("FAILED")
    return False


@run_priority(level="P0")
def test_auto_cost_base():
    """test_auto_cost_base"""
    cost = cost_model.Cost(memory=100, flops=200, time=0.5)
    assert check_cost(cost) is True
    print("test_auto_cost_base ... ok")


def test_auto_cost_comp():
    """test_auto_cost_comp"""
    x = paddle.static.data(name="x", shape=[20, 20], dtype="float32")
    y = paddle.static.data(name="y", shape=[20, 20], dtype="float32")

    z = paddle.matmul(x, y)
    print(z)

    matmul_v2_op = None
    ops = paddle.static.default_main_program().global_block().ops
    for op in ops:
        if op.type == "matmul_v2":
            matmul_v2_op = op
            break
    matmul_v2_cost = cost_model._g_op_cost_factory["matmul_v2"](op=matmul_v2_op)
    desc = build_comp_desc_from_op(op=matmul_v2_op)
    desc_str = build_comp_desc_str_for_predict(desc)
    print(desc_str)
    assert desc_str is not None
    assert check_cost(matmul_v2_cost.cost) is True

    time = calc_time_by_modeling(op=matmul_v2_op)
    assert time == matmul_v2_cost.cost.time
    tensor_cost = cost_model.TensorCost(tensor=x)
    # check memory
    assert tensor_cost.cost.memory == 1600
    print("test_auto_cost_comp ... ok")


def test_auto_cost_comm():
    """test_auto_cost_comm"""
    # Build cluster
    temp_dir = tempfile.TemporaryDirectory()
    cluster_json_path = os.path.join(temp_dir.name, "auto_parallel_cluster.json")
    cluster_json_object = json.loads(cluster_json)
    with open(cluster_json_path, "w") as cluster_json_file:
        json.dump(cluster_json_object, cluster_json_file)
    cluster = Cluster()
    cluster.build_from_file(cluster_json_path)

    # Build CommConetxt
    CommContext._has_instance = None
    CommContext._instance = None
    desc = {}
    desc["op"] = "c_allreduce_sum"
    desc["inputs"] = {"X": [(paddle.float32, [100, 200])]}
    desc["group_ranks"] = [0, 1]
    allreduce_cost = cost_model._g_op_cost_factory["c_allreduce_sum"](op_desc=desc, comm_context=CommContext(cluster))
    assert check_cost(allreduce_cost.cost) is True

    # Remove unnecessary files
    if os.path.exists(cluster_json_path):
        os.remove(cluster_json_path)

    temp_dir.cleanup()
    print("test_auto_cost_comm ... ok")


def test_auto_cost_estimator():
    """test_auto_cost_estimator"""
    # Build cluster
    temp_dir = tempfile.TemporaryDirectory()
    cluster_json_path = os.path.join(temp_dir.name, "auto_parallel_cluster.json")
    cluster_json_object = json.loads(cluster_json)
    with open(cluster_json_path, "w") as cluster_json_file:
        json.dump(cluster_json_object, cluster_json_file)
    cluster = Cluster()
    cluster.build_from_file(cluster_json_path)

    train_program = paddle.static.Program()
    cost_estimator = cost_model.CostEstimator(train_program, cluster=cluster)
    assert cost_estimator is not None

    # Remove unnecessary files
    if os.path.exists(cluster_json_path):
        os.remove(cluster_json_path)

    temp_dir.cleanup()
    print("test_auto_cost_estimator ... ok")


if __name__ == "__main__":
    test_auto_cost_base()
    test_auto_cost_comp()
    test_auto_cost_comm()
    test_auto_cost_estimator()
