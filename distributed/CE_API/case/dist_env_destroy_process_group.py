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
  * @file dist_env_destroy_process_group.py
  * @author liujie44@baidu.com
  * @date 2023-08-03 14:56
  * @brief
  *
  **************************************************************************/
"""
import sys
import numpy as np
import paddle
import paddle.distributed as dist


def test_destroy_process_group():
    """test_destroy_process_group"""
    dist.init_parallel_env()
    group = dist.new_group([0, 1])

    dist.destroy_process_group(group)
    dist.is_initialized()
    assert dist.is_initialized() is True

    dist.destroy_process_group()
    dist.is_initialized()
    assert dist.is_initialized() is False
    print("test_destroy_process_group... ok")


if __name__ == "__main__":
    test_destroy_process_group()
