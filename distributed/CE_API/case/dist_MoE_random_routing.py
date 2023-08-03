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
  * @file dist_MoE_random_routing.py
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


def random_routing(topk_idx, topk_value, prob, topk=2):
    """assert_compare"""
    if topk == 2:
        new_topk_idx = np.copy(topk_idx)
        for i in range(len(topk_idx)):
            val = topk_value[i][1]
            if val * 2 < prob[i]:
                new_topk_idx[i][1] = -1
        return new_topk_idx
    else:
        raise RuntimeError("only topk=2 is supported now")


class TestRandomRoutingAPIFp32:
    """TestRandomRoutingAPIFp32"""

    def setUp(self):
        """setUp"""
        self.dtype = "float32"
        self.init()

    def init(self):
        """init"""
        self.upper_range = 8
        self.x = np.random.randint(-1, self.upper_range, size=(200, 2)).astype("int64")
        self.prob = np.random.random((self.x.shape[0],)).astype(self.dtype)
        self.topk_value = np.random.random(self.x.shape).astype(self.dtype)
        self.out = random_routing(self.x, self.topk_value, self.prob).astype(self.dtype)
        self.place = paddle.CUDAPlace(0)

    @run_priority(level="P0")
    def test_MoE_random_routing_dygraph(self):
        """test_MoE_random_routing_dygraph"""
        paddle.disable_static()
        x = paddle.to_tensor(self.x)
        value = paddle.to_tensor(self.topk_value)
        prob = paddle.to_tensor(self.prob)
        out = utils._random_routing(x, value, prob)
        assert np.allclose(out.numpy(), self.out)


class TestRandomRoutingAPIFp16(TestRandomRoutingAPIFp32):
    """TestRandomRoutingAPIFp16"""

    def setUp(self):
        self.dtype = "float16"
        self.init()


if __name__ == "__main__":
    # test fp32
    test_instance32 = TestRandomRoutingAPIFp32()
    test_instance32.setUp()
    test_instance32.test_MoE_random_routing_dygraph()
    print("test_MoE_random_routing_dygraph fp32 passed")

    # test fp16
    test_instance16 = TestRandomRoutingAPIFp16()
    test_instance16.setUp()
    test_instance16.test_MoE_random_routing_dygraph()
    print("test_MoE_random_routing_dygraph fp16 passed")
