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
  * @file dist_MoE_prune_gate_by_capacity.py
  * @author liujie44@baidu.com
  * @date 2022-04-15 11:00
  * @brief
  *
  **************************************************************************/
"""
import numpy as np
import paddle
from paddle.distributed.models.moe import utils

from utils import run_priority


def count(x, upper_num):
    """count"""
    res = np.zeros((upper_num,)).astype(int)
    for i in x.reshape(-1):
        if i >= 0 and i < len(res):
            res[i] += 1
    return res


def limit_by_capacity(expert_count, _capacity, n_worker):
    """limit_by_capacity"""
    capacity = np.copy(_capacity)
    old_shape = expert_count.shape
    expert_count = np.reshape(expert_count, (n_worker, len(capacity)))
    output = np.zeros_like(expert_count)
    length = len(expert_count)
    for wid in range(length):
        for eid in range(len(expert_count[wid])):
            last_cap = capacity[eid]
            if last_cap >= 0:
                capacity[eid] -= expert_count[wid][eid]
            if last_cap >= expert_count[wid][eid]:
                output[wid][eid] = expert_count[wid][eid]
            elif last_cap >= 0:
                output[wid][eid] = last_cap
    return output.reshape(old_shape)


def prune_gate_by_capacity(gate_idx, expert_count, n_expert, n_worker):
    """assert_compare"""
    new_gate_idx = np.copy(gate_idx)
    expert_count = np.copy(expert_count)
    length = len(gate_idx)
    for i in range(length):
        idx = gate_idx[i]
        last_cap = expert_count[idx]
        if last_cap > 0:
            expert_count[idx] -= 1
        else:
            new_gate_idx[i] = -1
    return new_gate_idx


def assert_allclose(output, expected, n_expert):
    """assert"""
    c1 = count(output, n_expert)
    c2 = count(expected, n_expert)
    assert np.allclose(c1, c2)


class TestPruneGateByCapacityAPI(object):
    """TestPruneGateByCapacityAPI"""

    def init_test_case(self):
        """init_test_case"""
        self.gate_idx = np.random.randint(0, self.n_expert, size=(200,)).astype(self.dtype)
        expert_count = count(self.gate_idx, self.n_expert * self.n_worker)
        capacity = np.random.randint(10, 200, size=(self.n_expert,))
        self.expert_count = limit_by_capacity(expert_count, capacity, self.n_worker).astype(self.dtype)
        self.out = prune_gate_by_capacity(self.gate_idx, self.expert_count, self.n_expert, self.n_worker).astype(
            self.dtype
        )
        self.place = paddle.CUDAPlace(0)

    def setUp(self):
        """setUp"""
        self.n_expert = 24
        self.n_worker = 2
        self.dtype = "int64"
        self.init_test_case()

    @run_priority(level="P0")
    def test_MoE_prune_gate_by_capacity_static(self):
        """test_MoE_prune_gate_by_capacity_static"""
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            gate_idx_tensor = paddle.static.data("GateIdx", shape=self.gate_idx.shape, dtype="int64")
            expert_count_tensor = paddle.static.data("ExpertCount", shape=self.expert_count.shape, dtype="int64")
            out = utils._prune_gate_by_capacity(gate_idx_tensor, expert_count_tensor, self.n_expert, self.n_worker)
            exe = paddle.static.Executor(self.place)
            res = exe.run(
                feed={
                    "GateIdx": self.gate_idx,
                    "ExpertCount": self.expert_count,
                },
                fetch_list=out,
            )
        assert_allclose(res[0], self.out, self.n_expert)
        print("test_MoE_prune_gate_by_capacity_static passed!")

    @run_priority(level="P0")
    def test_MoE_prune_gate_by_capacity_dygraph(self):
        """test_MoE_prune_gate_by_capacity_dygraph"""
        paddle.disable_static(self.place)
        gate_idx_tensor = paddle.to_tensor(self.gate_idx)
        expert_count_tensor = paddle.to_tensor(self.expert_count)
        out = utils._prune_gate_by_capacity(gate_idx_tensor, expert_count_tensor, self.n_expert, self.n_worker)
        assert_allclose(out.numpy(), self.out, self.n_expert)
        print("test_MoE_prune_gate_by_capacity_dygraph passed!")


if __name__ == "__main__":
    test_instance = TestPruneGateByCapacityAPI()
    test_instance.setUp()
    test_instance.test_MoE_prune_gate_by_capacity_static()
    test_instance.test_MoE_prune_gate_by_capacity_dygraph()
