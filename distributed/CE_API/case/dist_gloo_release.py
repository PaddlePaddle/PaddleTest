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
  * @file dist_gloo_release.py
  * @author liyang109@baidu.com
  * @date 2020-11-17 16:30
  * @brief
  *
  **************************************************************************/
"""
import multiprocessing
from contextlib import closing
import socket
import paddle

from utils import run_priority

port_set = set()


def find_free_port():
    """find_free_port"""

    def _free_port():
        """_free_port"""
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.bind(("", 0))
            return s.getsockname()[1]

    while True:
        port = _free_port()
        if port not in port_set:
            port_set.add(port)
            return port


@run_priority(level="P0")
def test_gloo_release(id, rank_num, server_endpoint):
    """test_gloo_release"""
    paddle.distributed.gloo_init_parallel_env(id, rank_num, server_endpoint)
    paddle.distributed.gloo_barrier()
    paddle.distributed.gloo_release()
    print("test_gloo_release...  ok")


@run_priority(level="P0")
def test_gloo_release_with_multiprocess(num_of_ranks):
    """test_gloo_release_with_multiprocess"""
    jobs = []
    server_endpoint = "127.0.0.1:%s" % (find_free_port())
    for id in range(num_of_ranks):
        p = multiprocessing.Process(target=test_gloo_release, args=(id, num_of_ranks, server_endpoint))
        jobs.append(p)
        p.start()
    for proc in jobs:
        proc.join()
    print("test_gloo_release_with_multiprocess...  ok")


if __name__ == "__main__":
    test_gloo_release_with_multiprocess(2)
