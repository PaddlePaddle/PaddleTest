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
  * @file dist_env_get_backend.py
  * @author liujie44@baidu.com
  * @date 2023-08-03 14:56
  * @brief
  *
  **************************************************************************/
"""
import sys
import numpy as np
import paddle


def test_get_backend():
    """test_get_backend"""
    paddle.distributed.init_parallel_env()
    paddle.distributed.get_backend()
    assert paddle.distributed.get_backend() == "NCCL"
    print("test_get_backend ... ok")


if __name__ == "__main__":
    test_get_backend()
