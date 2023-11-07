#!/bin/env python3
# -*- coding: utf-8 -*-
# @author Zeref996
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
runner
"""

import os
import socket
import platform
import traceback

import argparse
import json
import sys
from datetime import datetime

from statistics.statistics import Statistics
from info.snapshot import Snapshot
from alarm.alarm import Alarm

sys.path.append("..")
from utils.yaml_loader import YamlLoader
from utils.logger import Logger
from benchtrans import BenchTrans
from jelly.jelly_v2 import Jelly_v2
from runner_base import ApiBenchmarkBASE

import paddle

import psutil

p = psutil.Process()
p.cpu_affinity([2])

SKIP_DICT = {"Windows": ["fft"], "Darwin": ["fft"], "Linux": []}
INDEX_DICT = {}
SPECIAL = False  # speacial for emergency


class ApiBenchmarkForUser(ApiBenchmarkBASE):
    """
    api benchmark 调度CI, 监控cpu+前向, 支持多个机器baseline
    """

    def __init__(self, yaml_path, framework, enable_backward, place, yaml_info):
        """
        :param baseline: 性能baseline键值对, key为case名, value为性能float
        """
        # 测试控制项
        self.experi = 1
        self.loops = 50  # 循环次数
        self.base_times = 1000  # timeit 基础运行时间
        self.default_dtype = "float32"
        self.if_showtime = True
        self.double_check = True
        self.check_iters = 5
        self.now_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # md5唯一标识码
        self.md5_id = Snapshot().get_md5_id()

        # 效率云环境变量
        self.AGILE_PIPELINE_BUILD_ID = os.environ.get("AGILE_PIPELINE_BUILD_ID", 0)
        self.description = {"success": True, "reason": "ok", "pipelineBuildId": self.AGILE_PIPELINE_BUILD_ID}

        # 例行标识
        self.ci = 0
        self.uid = -1

        # 框架信息
        self.framework = framework

        # 框架信息callback
        if self.framework == "torch":
            import torch

            self.commit = "torch_commit"
            self.version = torch.__version__
            self.cuda = torch.version.cuda
            self.cudnn = torch.backends.cudnn.version()
        else:
            self.commit = paddle.__git_commit__
            self.version = paddle.__version__
            self.cuda = paddle.version.cuda()
            self.cudnn = paddle.version.cudnn()

        # 获取所有case名称
        self.yaml_path = yaml_path
        self.yaml_loader = YamlLoader(self.yaml_path)
        self.all_cases = self.yaml_loader.get_all_case_name()

        # 项目配置信息
        self.place = place
        self.enable_backward = enable_backward
        self.yaml_info = yaml_info
        self.card = 0

        # 机器系统信息
        self.hostname = socket.gethostname()
        self.system = platform.system()
        self.snapshot = {
            "os": platform.platform(),
            "card": self.card,
            "cuda": self.cuda,
            "cudnn": self.cudnn,
        }

        # 初始化日志
        self.logger = Logger("ApiBenchmarkPTS")

        # 初始化统计模块
        self.statistics = Statistics()

        # 邮件报警
        # self.email = Alarm(storage=self.storage)

    def _run_ci(self):
        """

        :return:
        """
        error_dict = self._run_main(all_cases=self.all_cases, loops=self.loops, base_times=self.base_times)
        print("error is: ", error_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    # yaml_path, routine, comment, framework, enable_backward, python, place, yaml_info, wheel_link
    parser.add_argument(
        "--yaml", type=str, default="./../yaml/ci_api_benchmark_fp32_cuda118_py310.yml", help="input the yaml path"
    )
    parser.add_argument("--framework", type=str, default="paddle", help="[paddle] | [torch]")
    parser.add_argument("--enable_backward", type=int, default=0, help="if 1, enable backward test")
    parser.add_argument("--place", type=str, default="cpu", help="[cpu] or [gpu]")
    parser.add_argument("--yaml_info", type=str, default="case_0", help="[case_0] or [case_1] or [case_2]")
    args = parser.parse_args()

    api_bm = ApiBenchmarkForUser(
        yaml_path=args.yaml,
        framework=args.framework,
        enable_backward=args.enable_backward,
        place=args.place,
        yaml_info=args.yaml_info,
    )
    api_bm._run_ci()
    # python runner_nodb.py --yaml broadcast_shape.yml --framework paddle
    # --enable_backward 0 --place cpu 2>&1 | tee -a f_broadcast_1w.log
