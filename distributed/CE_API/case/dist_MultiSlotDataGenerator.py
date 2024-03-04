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
  * @file dist_MultiSlotDataGenerator.py
  * @author liujie44@baidu.com
  * @date 2024-02-20
  * @brief
  *
  **************************************************************************/
"""
import paddle
import paddle.distributed.fleet.data_generator as dg
from utils import run_priority

paddle.enable_static()


class MyData(dg.DataGenerator):
    """MyData"""

    def generate_sample(self, line):
        """generate_sample"""

        def local_iter():
            yield ("words", [1, 2, 3, 4])

        return local_iter

    def generate_batch(self, samples):
        """generate_batch"""

        def local_iter():
            for s in samples:
                yield ("words", s[1].extend([s[1][0]]))


@run_priority(level="P0")
def test_MultiSlotDataGenerator_generate_batch():
    """test_MultiSlotDataGenerator_generate_batch"""
    mydata = MyData()
    mydata.generate_batch([1, 2, 3, 4])

    print("test_MultiSlotDataGenerator_generate_batch ... ok")


@run_priority(level="P0")
def test_MultiSlotDataGenerator_generate_sample():
    """test_MultiSlotDataGenerator_generate_sample"""
    mydata = MyData()
    mydata.generate_sample([1, 2, 3, 4])

    print("test_MultiSlotDataGenerator_generate_sample ... ok")


def test_MultiSlotDataGenerator_set_batch():
    """test_MultiSlotDataGenerator_set_batch"""
    mydata = MyData()
    mydata.set_batch(128)

    print("test_MultiSlotDataGenerator_set_batch ... ok")


if __name__ == "__main__":
    test_MultiSlotDataGenerator_generate_batch()
    test_MultiSlotDataGenerator_generate_sample()
    test_MultiSlotDataGenerator_set_batch()
