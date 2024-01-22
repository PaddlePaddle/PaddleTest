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
  * @file dist_fleet_utilbase_get_file_shard.py
  * @author liujie44@baidu.com
  * @date 2021-11-22 19:40
  * @brief
  *
  **************************************************************************/
"""
import paddle.distributed.fleet as fleet
import paddle.distributed.fleet.base.role_maker as role_maker
from utils import run_priority


@run_priority(level="P0")
def test_file_shard():
    """test utilbase file shard"""
    role = role_maker.UserDefinedRoleMaker(
        is_collective=False,
        init_gloo=False,
        current_id=0,
        role=role_maker.Role.WORKER,
        worker_endpoints=["127.0.0.1:6003", "127.0.0.1:6004"],
        server_endpoints=["127.0.0.1:6001", "127.0.0.1:6002"],
    )
    fleet.init(role)

    files = fleet.util.get_file_shard(["file1", "file2", "file3"])
    print(files)
    assert len(files) == 2


if __name__ == "__main__":
    test_file_shard()
