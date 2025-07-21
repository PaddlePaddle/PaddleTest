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
  * Copyright (c) 2025 Baidu.com, Inc. All Rights Reserved
  * @file dist_P2POp.py
  * @author lvkunpeng@baidu.com
  * @date 2025-07-14
  * @brief
  *
  **************************************************************************/
"""
import paddle
import pytest
import paddle.distributed as dist
from paddle.distributed import P2POp
from utils import run_priority

dist.init_parallel_env()

@run_priority(level="P0")
def test_P2POp_construction():
    """test_P2POp_construction"""
    send_tensor = paddle.arange(4, dtype='float32')
    recv_tensor = paddle.empty(shape=[4], dtype='float32')
    peer_rank = 1

    send_op = P2POp(dist.isend, send_tensor, peer_rank)
    recv_op = P2POp(dist.irecv, recv_tensor, peer_rank)

    assert send_op.op == dist.isend
    assert send_op.tensor is send_tensor
    assert send_op.peer == peer_rank
    assert recv_op.op == dist.irecv
    assert recv_op.tensor is recv_tensor
    assert recv_op.peer == peer_rank

    print("test_P2POp_construction ... ok")

@run_priority(level="P0")
def test_P2POp_invalid_op():
    """test_P2POp_invalid_op"""
    with pytest.raises(RuntimeError, match="Invalid ``op`` function"):
        P2POp(lambda x: x, paddle.arange(2), 0)
    print("test_P2POp_invalid_op ... ok")

@run_priority(level="P0")
def test_P2POp_tensor_wrong_type():
    """test_P2POp_tensor_wrong_type"""
    with pytest.raises(ValueError): 
        op = P2POp(dist.isend, None, 0)
        dist.batch_isend_irecv([op])  
    print("test_P2POp_tensor_wrong_type ... ok")

@run_priority(level="P0")
def test_P2POp_group_none_vs_explicit():
    """test_P2POp_group_none_vs_explicit"""
    tensor = paddle.arange(2)
    op1 = P2POp(dist.isend, tensor, 1)
    op2 = P2POp(dist.isend, tensor, 1, group=None)
    assert op1.group is not None
    assert op2.group is not None
    assert op1.group == op2.group
    print("test_P2POp_group_none_vs_explicit ... ok")

@run_priority(level="P0")
def test_P2POp_with_different_groups():
    """test_P2POp_with_different_groups"""
    ranks = [0, 1]
    group = dist.new_group(ranks=ranks)
    tensor = paddle.arange(2)
    op = P2POp(dist.isend, tensor, 1, group=group)
    assert op.group == group
    print("test_P2POp_with_different_groups ... ok")


if __name__ == "__main__":
    test_P2POp_construction()
    test_P2POp_invalid_op()
    test_P2POp_tensor_wrong_type()
    test_P2POp_group_none_vs_explicit()
    test_P2POp_with_different_groups()