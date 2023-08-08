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
  * @file dist_auto_cluster.py
  * @author liujie44@baidu.com
  * @date 2022-09-07 11:00
  * @brief
  *
  **************************************************************************/
"""
import json
import os
import tempfile

from paddle.distributed.auto_parallel.static.cluster import (
    Cluster,
    get_default_cluster,
)

cluster_json = """
{
    "alpha_latency": {"inter": {"ring": "NET", "tree": "NET"},
                    "intra": {"ring": "NVL", "tree": "PHB"},
                    "base": {"ring": 8.4, "tree": 0},
                    "switch": 10.0},
    "machines": [
        {
            "hostname": "yq01-sys-hic-v100-box-a225-0266",
            "addr": "10.127.9.147",
            "port": "60009",
            "devices": [
                {
                    "global_id": 0,
                    "local_id": 0,
                    "type": "GPU",
                    "model": " Tesla V100-SXM2-32GB",
                    "memory": "32",
                    "sp_gflops": "15700",
                    "dp_gflops": "7800"
                },
                {
                    "global_id": 1,
                    "local_id": 1,
                    "type": "GPU",
                    "model": " Tesla V100-SXM2-32GB",
                    "memory": "32",
                    "sp_gflops": "15700",
                    "dp_gflops": "7800"
                },
                {
                    "global_id": 2,
                    "local_id": 2,
                    "type": "GPU",
                    "model": " Tesla V100-SXM2-32GB",
                    "memory": "32",
                    "sp_gflops": "15700",
                    "dp_gflops": "7800"
                },
                {
                    "global_id": 3,
                    "local_id": 3,
                    "type": "GPU",
                    "model": " Tesla V100-SXM2-32GB",
                    "memory": "32",
                    "sp_gflops": "15700",
                    "dp_gflops": "7800"
                },
                {
                    "global_id": 4,
                    "local_id": 4,
                    "type": "GPU",
                    "model": " Tesla V100-SXM2-32GB",
                    "memory": "32",
                    "sp_gflops": "15700",
                    "dp_gflops": "7800"
                },
                {
                    "global_id": 5,
                    "local_id": 5,
                    "type": "GPU",
                    "model": " Tesla V100-SXM2-32GB",
                    "memory": "32",
                    "sp_gflops": "15700",
                    "dp_gflops": "7800"
                },
                {
                    "global_id": 6,
                    "local_id": 6,
                    "type": "GPU",
                    "model": " Tesla V100-SXM2-32GB",
                    "memory": "32",
                    "sp_gflops": "15700",
                    "dp_gflops": "7800"
                },
                {
                    "global_id": 7,
                    "local_id": 7,
                    "type": "GPU",
                    "model": " Tesla V100-SXM2-32GB",
                    "memory": "32",
                    "sp_gflops": "15700",
                    "dp_gflops": "7800"
                },
                {
                    "global_id": 8,
                    "local_id": 0,
                    "type": "CPU",
                    "arch": "x86_64",
                    "vendor": "GenuineIntel",
                    "model": "Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GH",
                    "memory": "502",
                    "sp_gflops": "150",
                    "dp_gflops": "75"
                },
                {
                    "global_id": 9,
                    "local_id": 0,
                    "type": "NIC",
                    "width": 12.5,
                    "ip": "10.127.9.147"
                }
            ],
            "links": [
                {
                    "source_global_id": 0,
                    "target_global_id": 1,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 0,
                    "target_global_id": 2,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 0,
                    "target_global_id": 3,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 0,
                    "target_global_id": 4,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 0,
                    "target_global_id": 5,
                    "type": "NVB",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 0,
                    "target_global_id": 6,
                    "type": "NVB",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 0,
                    "target_global_id": 7,
                    "type": "NVB",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 0,
                    "target_global_id": 8,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 0,
                    "target_global_id": 9,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 1,
                    "target_global_id": 0,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 1,
                    "target_global_id": 2,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 1,
                    "target_global_id": 3,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 1,
                    "target_global_id": 4,
                    "type": "NVB",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 1,
                    "target_global_id": 5,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 1,
                    "target_global_id": 6,
                    "type": "NVB",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 1,
                    "target_global_id": 7,
                    "type": "NVB",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 1,
                    "target_global_id": 8,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 1,
                    "target_global_id": 9,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 2,
                    "target_global_id": 0,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 2,
                    "target_global_id": 1,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 2,
                    "target_global_id": 3,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 2,
                    "target_global_id": 4,
                    "type": "NVB",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 2,
                    "target_global_id": 5,
                    "type": "NVB",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 2,
                    "target_global_id": 6,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 2,
                    "target_global_id": 7,
                    "type": "NVB",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 2,
                    "target_global_id": 8,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 2,
                    "target_global_id": 9,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 3,
                    "target_global_id": 0,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 3,
                    "target_global_id": 1,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 3,
                    "target_global_id": 2,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 3,
                    "target_global_id": 4,
                    "type": "NVB",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 3,
                    "target_global_id": 5,
                    "type": "NVB",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 3,
                    "target_global_id": 6,
                    "type": "NVB",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 3,
                    "target_global_id": 7,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 3,
                    "target_global_id": 8,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 3,
                    "target_global_id": 9,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 4,
                    "target_global_id": 0,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 4,
                    "target_global_id": 1,
                    "type": "NVB",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 4,
                    "target_global_id": 2,
                    "type": "NVB",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 4,
                    "target_global_id": 3,
                    "type": "NVB",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 4,
                    "target_global_id": 5,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 4,
                    "target_global_id": 6,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 4,
                    "target_global_id": 7,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 4,
                    "target_global_id": 8,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 4,
                    "target_global_id": 9,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 5,
                    "target_global_id": 0,
                    "type": "NVB",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 5,
                    "target_global_id": 1,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 5,
                    "target_global_id": 2,
                    "type": "NVB",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 5,
                    "target_global_id": 3,
                    "type": "NVB",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 5,
                    "target_global_id": 4,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 5,
                    "target_global_id": 6,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 5,
                    "target_global_id": 7,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 5,
                    "target_global_id": 8,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 5,
                    "target_global_id": 9,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 6,
                    "target_global_id": 0,
                    "type": "NVB",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 6,
                    "target_global_id": 1,
                    "type": "NVB",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 6,
                    "target_global_id": 2,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 6,
                    "target_global_id": 3,
                    "type": "NVB",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 6,
                    "target_global_id": 4,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 6,
                    "target_global_id": 5,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 6,
                    "target_global_id": 7,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 6,
                    "target_global_id": 8,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 6,
                    "target_global_id": 9,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 7,
                    "target_global_id": 0,
                    "type": "NVB",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 7,
                    "target_global_id": 1,
                    "type": "NVB",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 7,
                    "target_global_id": 2,
                    "type": "NVB",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 7,
                    "target_global_id": 3,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 7,
                    "target_global_id": 4,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 7,
                    "target_global_id": 5,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 7,
                    "target_global_id": 6,
                    "type": "NVL",
                    "bandwidth": 235.0
                },
                {
                    "source_global_id": 7,
                    "target_global_id": 8,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 7,
                    "target_global_id": 9,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 8,
                    "target_global_id": 0,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 8,
                    "target_global_id": 1,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 8,
                    "target_global_id": 2,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 8,
                    "target_global_id": 3,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 8,
                    "target_global_id": 4,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 8,
                    "target_global_id": 5,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 8,
                    "target_global_id": 6,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 8,
                    "target_global_id": 7,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 8,
                    "target_global_id": 9,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 9,
                    "target_global_id": 0,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 9,
                    "target_global_id": 1,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 9,
                    "target_global_id": 2,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 9,
                    "target_global_id": 3,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 9,
                    "target_global_id": 4,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 9,
                    "target_global_id": 5,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 9,
                    "target_global_id": 6,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 9,
                    "target_global_id": 7,
                    "type": "PHB",
                    "bandwidth": 24.0
                },
                {
                    "source_global_id": 9,
                    "target_global_id": 8,
                    "type": "PHB",
                    "bandwidth": 24.0
                }
            ]
        }
    ]
}
"""
