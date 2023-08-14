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
  * @file dist_MoE_assign_pos.py
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


def assign_pos(x, _cum_count):
    """assert_compare"""
    cum_count = np.copy(_cum_count)
    x = x.reshape(-1)
    res = np.zeros((cum_count[-1],), dtype=np.int64)
    for i, idx in enumerate(x):
        p = cum_count[idx]
        cum_count[idx] -= 1
        if p >= 1:
            res[p - 1] = i
    return res


def count(x, upper_num):
    """assert_count"""
    res = np.zeros((upper_num,)).astype(int)
    for i in x.reshape(-1):
        if i >= 0 and i < len(res):
            res[i] += 1
    return res


def assert_allclose(res, out, cum_count):
    """assert"""
    c0 = 0
    for c in cum_count:
        if c == c0:
            continue
        data1 = np.copy(res[c0:c])
        data2 = np.copy(out[c0:c])
        data1.sort()
        data2.sort()
        assert np_allclose(data2, data1)
        c0 = c
    return True


np_allclose = np.allclose


class TestAssignPosAPI(object):
    """TestAssignPosAPI"""

    def setUp(self):
        """setUp"""
        self.x = np.random.randint(0, 16, size=(100, 2)).astype("int64")
        y = count(self.x, 16)
        self.cum_count = np.cumsum(y).astype(self.x.dtype)
        self.out = assign_pos(self.x, self.cum_count)
        self.place = paddle.CUDAPlace(0)

    @run_priority(level="P0")
    def test_MoE_assign_pos_static(self):
        """test_MoE_assign_pos_static"""
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data("x", self.x.shape, dtype="int64")
            cum_count = paddle.static.data("cum_count", self.cum_count.shape, dtype="int64")
            out = utils._assign_pos(x, cum_count)
            exe = paddle.static.Executor(self.place)
            res = exe.run(feed={"x": self.x, "cum_count": self.cum_count}, fetch_list=[out])
            assert_allclose(res[0], self.out, self.cum_count)
            print("test_MoE_assign_pos_static passed!")

    @run_priority(level="P0")
    def test_MoE_assign_pos_dygraph(self):
        """test_MoE_assign_pos_dygraph"""
        paddle.disable_static()
        x = paddle.to_tensor(self.x)
        cum_count = paddle.to_tensor(self.cum_count).astype(x.dtype)

        out = utils._assign_pos(x, cum_count)
        assert_allclose(out.numpy(), self.out, self.cum_count)
        print("test_MoE_assign_pos_dygraph passed!")


if __name__ == "__main__":
    test_instance = TestAssignPosAPI()
    test_instance.setUp()
    test_instance.test_MoE_assign_pos_static()
    test_instance.test_MoE_assign_pos_dygraph()
