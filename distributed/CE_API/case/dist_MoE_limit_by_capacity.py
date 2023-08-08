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
  * @file dist_MoE_limit_by_capacity.py
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


def limit_by_capacity(expert_count, _capacity, n_worker):
    """ "assert_compare"""
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


def all_close(exp, out, n_worker):
    """assert"""
    exp = exp.reshape(n_worker, -1)
    out = out.reshape(n_worker, -1)
    return np.allclose(exp.sum(0), out.sum(0))


class TestLimitByCapacityAPI(object):
    """TestLimitByCapacityAPI"""

    def init_test_case(self):
        """init_test_case"""
        self.expert_count = np.random.randint(0, 1000, size=(len(self.capacity) * self.n_worker))
        self.out = limit_by_capacity(self.expert_count, self.capacity, self.n_worker)
        self.expert_count = self.expert_count.astype("int64")
        self.capacity = self.capacity.astype("int64")
        self.place = paddle.CUDAPlace(0)

    def setUp(self):
        """setUp"""
        self.capacity = np.array([100, 12000, 1200, 800, 4700, 10000, 57, 99])
        self.n_worker = 1024 * 8
        self.init_test_case()

    @run_priority(level="P0")
    def test_MoE_limit_by_capacity_static(self):
        """test_MoE_limit_by_capacity_static"""
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            capacity = paddle.static.data("capacity", shape=self.capacity.shape, dtype="int64")
            expert_count_tensor = paddle.static.data("ExpertCount", shape=self.expert_count.shape, dtype="int64")
            out = utils._limit_by_capacity(expert_count_tensor, capacity, self.n_worker)
            exe = paddle.static.Executor(self.place)
            res = exe.run(
                feed={
                    "capacity": self.capacity,
                    "ExpertCount": self.expert_count,
                },
                fetch_list=out,
            )

        assert all_close(self.out, res[0], self.n_worker)
        print("test_MoE_limit_by_capacity_static passed!")

    @run_priority(level="P0")
    def test_MoE_limit_by_capacity_dygraph(self):
        """test_limit_by_capacity_dygraph"""
        paddle.disable_static(self.place)
        capacity = paddle.to_tensor(self.capacity)
        expert_count_tensor = paddle.to_tensor(self.expert_count)
        out = utils._limit_by_capacity(expert_count_tensor, capacity, self.n_worker)
        assert all_close(self.out, out.numpy(), self.n_worker)
        print("test_MoE_limit_by_capacity_dygraph passed!")


if __name__ == "__main__":
    test_instance = TestLimitByCapacityAPI()
    test_instance.setUp()
    test_instance.test_MoE_limit_by_capacity_static()
    test_instance.test_MoE_limit_by_capacity_dygraph()
