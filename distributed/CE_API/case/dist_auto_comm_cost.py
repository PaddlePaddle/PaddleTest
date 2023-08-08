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
  * @file dist_auto_comm_cost.py
  * @author liujie44@baidu.com
  * @date 2022-09-07 11:00
  * @brief
  *
  **************************************************************************/
"""
import json
import os
import tempfile

import paddle
from paddle.distributed.auto_parallel.static.cluster import Cluster
from paddle.distributed.auto_parallel.static.cost import (
    AllgatherOpCost,
    AllreduceSumOpCost,
    BroadcastOpCost,
    CommContext,
    IdentityOpCost,
    RecvOpCost,
    SendOpCost,
    build_comm_desc,
)

from dist_auto_cluster import cluster_json
from dist_auto_multi_cluster import multi_cluster_json
from utils import run_priority


@run_priority(level="P0")
def test_comm_cost():
    """test_comm_cost"""
    # setUp
    temp_dir = tempfile.TemporaryDirectory()

    # Build cluster
    cluster_json_path = os.path.join(temp_dir.name, "auto_parallel_cluster0.json")
    cluster_json_object = json.loads(cluster_json)
    with open(cluster_json_path, "w") as cluster_json_file:
        json.dump(cluster_json_object, cluster_json_file)
    cluster = Cluster()
    cluster.build_from_file(cluster_json_path)

    # Build CommConetxt
    CommContext._has_instance = None
    CommContext._instance = None
    comm_context = CommContext(cluster)

    # Check AllreduceSumCost 128MB ring cost
    allreduce_sum_op_desc = build_comm_desc(
        "c_allreduce_sum",
        [0, 1, 2, 3, 4, 5, 6, 7],
        paddle.float32,
        [1, 32 * (10**6)],
    )
    allreduce_sum_op_cost = AllreduceSumOpCost(op_desc=allreduce_sum_op_desc, comm_context=comm_context)
    assert allreduce_sum_op_cost.time > 0

    # Check AllgatherOpCost cost
    allgather_op_desc = build_comm_desc(
        "c_allgather",
        [0, 1, 2, 3, 4, 5, 6, 7],
        paddle.float32,
        [1, 32 * (10**6)],
    )
    allgather_op_cost = AllgatherOpCost(op_desc=allgather_op_desc, comm_context=comm_context)
    assert allgather_op_cost.time > 0

    # Check BroadcastOpCost cost
    broadcast_op_desc = build_comm_desc(
        "c_broadcast",
        [0, 1, 2, 3, 4, 5, 6, 7],
        paddle.float32,
        [1, 32 * (10**6)],
    )
    broadcast_op_cost = BroadcastOpCost(op_desc=broadcast_op_desc, comm_context=comm_context)
    assert broadcast_op_cost.time > 0

    # Check SendOpCost cost
    send_op_desc = build_comm_desc("send_v2", [0, 1], paddle.float32, [1, 32 * (10**6)])
    send_op_cost = SendOpCost(op_desc=send_op_desc, comm_context=comm_context)
    assert send_op_cost.time > 0

    # Check RecvOpCost cost
    recv_op_desc = build_comm_desc("recv_v2", [0, 1], paddle.float32, [1, 32 * (10**6)])
    recv_op_cost = RecvOpCost(op_desc=recv_op_desc, comm_context=comm_context)
    assert recv_op_cost.time > 0

    # Check IdentityOpCost cost
    identity_op_desc = build_comm_desc("c_identity", [0, 1], paddle.float32, [1, 32 * (10**6)])
    identity_op_cost = IdentityOpCost(op_desc=identity_op_desc, comm_context=comm_context)
    assert identity_op_cost.time >= 0

    temp_dir.cleanup()
    print("test_comm_cost ... ok")


@run_priority(level="P0")
def test_cross_machine_comm_cost():
    """test_cross_machine_comm_cost"""
    # setUp
    temp_dir = tempfile.TemporaryDirectory()

    # Build cluster
    cluster_json_path = os.path.join(temp_dir.name, "auto_parallel_cluster1.json")
    cluster_json_object = json.loads(multi_cluster_json)
    with open(cluster_json_path, "w") as cluster_json_file:
        json.dump(cluster_json_object, cluster_json_file)
    cluster = Cluster()
    cluster.build_from_file(cluster_json_path)

    # Build CommConetxt
    CommContext._has_instance = None
    CommContext._instance = None
    comm_context = CommContext(cluster)

    # Check AllreduceSumCost 128MB ring cost
    allreduce_sum_op_desc = build_comm_desc(
        "c_allreduce_sum",
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        paddle.float32,
        [1, 32 * (10**6)],
    )
    allreduce_sum_op_cost = AllreduceSumOpCost(op_desc=allreduce_sum_op_desc, comm_context=comm_context)
    assert allreduce_sum_op_cost.time > 0

    # Check AllgatherOpCost cost
    allgather_op_desc = build_comm_desc(
        "c_allgather",
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        paddle.float32,
        [1, 32 * (10**6)],
    )
    allgather_op_cost = AllgatherOpCost(op_desc=allgather_op_desc, comm_context=comm_context)
    assert allgather_op_cost.time > 0

    # Check BroadcastOpCost cost
    broadcast_op_desc = build_comm_desc(
        "c_broadcast",
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        paddle.float32,
        [1, 32 * (10**6)],
    )
    broadcast_op_cost = BroadcastOpCost(op_desc=broadcast_op_desc, comm_context=comm_context)
    assert broadcast_op_cost.time > 0

    # Check SendOpCost cost
    send_op_desc = build_comm_desc("send_v2", [0, 1], paddle.float32, [1, 32 * (10**6)])
    send_op_cost = SendOpCost(op_desc=send_op_desc, comm_context=comm_context)
    assert send_op_cost.time > 0

    # Check RecvOpCost cost
    recv_op_desc = build_comm_desc("recv_v2", [0, 1], paddle.float32, [1, 32 * (10**6)])
    recv_op_cost = RecvOpCost(op_desc=recv_op_desc, comm_context=comm_context)
    assert recv_op_cost.time > 0

    # Remove unnecessary files
    if os.path.exists(cluster_json_path):
        os.remove(cluster_json_path)

    temp_dir.cleanup()
    print("test_cross_machine_comm_cost ... ok")


if __name__ == "__main__":
    test_comm_cost()
    test_cross_machine_comm_cost()
