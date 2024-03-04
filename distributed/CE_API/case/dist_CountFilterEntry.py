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
  * @file dist_CountFilterEntry.py
  * @author liujie44@baidu.com
  * @date 2024-02-20
  * @brief
  *
  **************************************************************************/
"""
import paddle
from utils import run_priority

paddle.enable_static()


@run_priority(level="P0")
def test_CountFilterEntry():
    """test_CountFilterEntry"""
    sparse_feature_dim = 1024
    embedding_size = 64

    entry = paddle.distributed.CountFilterEntry(10)

    input = paddle.static.data(name="ins", shape=[1], dtype="int64")

    emb = paddle.static.nn.sparse_embedding(
        input=input,
        size=[sparse_feature_dim, embedding_size],
        is_test=False,
        entry=entry,
        param_attr=paddle.ParamAttr(name="SparseFeatFactors", initializer=paddle.nn.initializer.Uniform()),
    )
    print(emb)
    print("test_CountFilterEntry ... ok")


if __name__ == "__main__":
    test_CountFilterEntry()
