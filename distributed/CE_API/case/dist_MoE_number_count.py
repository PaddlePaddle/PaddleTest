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
  * @file dist_MoE_number_count.py
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
    """assert_count"""
    res = np.zeros((upper_num,)).astype(int)
    for i in x.reshape(-1):
        if i >= 0 and i < len(res):
            res[i] += 1
    return res


class TestNumberCountAPI(object):
    """TestNumberCountAPI"""

    def setUp(self):
        """setUp"""
        self.upper_num = 320
        self.x = np.random.randint(-1, self.upper_num, size=(6000, 200)).astype("int64")
        self.out = count(self.x, self.upper_num)
        self.place = paddle.CUDAPlace(0)

    @run_priority(level="P0")
    def test_MoE_number_count_static(self):
        """test_MoE_number_count_static"""
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data("x", self.x.shape, dtype="int64")
            out = utils._number_count(x, self.upper_num)
            exe = paddle.static.Executor(self.place)
            res = exe.run(feed={"x": self.x}, fetch_list=[out])
            assert np.allclose(res, self.out)
        print("test_MoE_number_count_static mode passed!")

    @run_priority(level="P0")
    def test_MoE_number_count_dygraph(self):
        """test_MoE_number_count_dygraph"""
        paddle.disable_static()
        x = paddle.to_tensor(self.x)
        out = utils._number_count(x, self.upper_num)
        assert np.allclose(out.numpy(), self.out)
        print("test_MoE_number_count_dygraph passed!")


if __name__ == "__main__":
    test_instance = TestNumberCountAPI()
    test_instance.setUp()
    test_instance.test_MoE_number_count_static()
    test_instance.test_MoE_number_count_dygraph()
