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
  * @file dist_gather.py
  * @author lvkunpeng@baidu.com
  * @date 2025-07-15
  * @brief
  *
  **************************************************************************/
"""
import numpy as np
import paddle
import paddle.distributed as dist
import pytest
from paddle.distributed import gather
from utils import run_priority

SUPPORTED_DTYPES = [np.float16, np.float32, np.float64, 
                    np.int32, np.int64, np.int8, 
                    np.uint8, paddle.bool, paddle.bfloat16]

dist.init_parallel_env()

# @pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
@run_priority(level="P0")
def test_gather(dtype):
    """test_gather"""
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if dtype == paddle.bool:
        data = paddle.to_tensor(
        [rank % 2 == 0, rank % 3 == 0, True], dtype=dtype
        )
    else:
        data = paddle.to_tensor(
        [rank + 1, rank + 2, rank + 3], dtype=dtype
    )

    gather_list = []
    dist.stream.gather(data, gather_list, dst=0)

    if rank == 0:
        assert len(gather_list) == world_size
        for idx, t in enumerate(gather_list):
            if dtype == paddle.bool:
                expected = paddle.to_tensor(
                    [idx % 2 == 0, idx % 3 == 0, True], dtype=dtype
                )
                assert paddle.all(paddle.equal(t, expected))
            else:
                expected = paddle.to_tensor(
                    [idx + 1, idx + 2, idx + 3], dtype=dtype
                )
                assert paddle.allclose(t.cast("float32"), expected.cast("float32"))
    else:
        assert gather_list == []
    
    print("test_gather ... ok")


if __name__ == "__main__":
    # pytest.main([__file__])
    for dtype in SUPPORTED_DTYPES:
        test_gather(dtype)