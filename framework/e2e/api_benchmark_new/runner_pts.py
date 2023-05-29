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

import argparse
import json
import sys
from datetime import datetime

from statistics.statistics import Statistics
from db.pts_db import PTSdb
from info.snapshot import Snapshot
from alarm.alarm import Alarm

import paddle

sys.path.append("..")
from utils.logger import Logger
from runner_base import ApiBenchmarkBASE


SKIP_DICT = {"Windows": ["fft"], "Darwin": ["fft"], "Linux": []}
INDEX_DICT = {}
SPECIAL = False  # speacial for emergency


class ApiBenchmarkPTS(ApiBenchmarkBASE):
    """
    api benchmark 调度CI, 监控cpu+前向, 支持多个机器baseline
    """

    def __init__(self, latest_id, yaml_path, comment, framework, enable_backward, python, place, yaml_info, wheel_link):
        super(ApiBenchmarkPTS, self).__init__(yaml_path)
        """
        :param baseline: 性能baseline键值对, key为case名, value为性能float
        """
        # 测试控制项
        self.loops = 50  # 循环次数
        self.base_times = 1000  # timeit 基础运行时间
        self.default_dtype = "float32"
        self.if_showtime = True
        self.double_check = False
        self.check_iters = 5
        self.now_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 初始化数据库
        self.db = PTSdb(storage=self.storage)

        # md5唯一标识码
        self.md5_id = Snapshot().get_md5_id()

        # 例行标识
        self.latest_id = latest_id
        self.comment = comment
        self.ci = 0

        # 框架信息
        self.framework = framework
        self.wheel_link = wheel_link

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

        # 项目配置信息
        self.place = place
        self.python = python
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
            "comment": self.comment,
        }

        # 初始化日志
        self.logger = Logger("ApiBenchmarkPTS")

        # 初始化统计模块
        self.statistics = Statistics()

        # 邮件报警
        # self.email = Alarm(storage=self.storage)

    def _run_pts(self):
        """

        :return:
        """
        job_dict = {
            # 'status': 'done',
            "commit": self.commit,
            "version": self.version,
            "hostname": self.hostname,
            "place": self.place,
            "system": self.system,
            "cuda": self.cuda,
            "cudnn": self.cudnn,
            "snapshot": json.dumps(self.snapshot),
            "md5_id": self.md5_id,
            # "uid": self.uid,
            # "routine": self.routine,
            "ci": self.ci,
            "comment": self.comment,
            "enable_backward": self.enable_backward,
            "python": self.python,
            "yaml_info": self.yaml_info,
            "wheel_link": self.wheel_link,
            # "description": json.dumps(self.description),
            # "create_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "update_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        error_dict = self._run_main(all_cases=self.all_cases, latest_id=self.latest_id)

        if bool(error_dict):
            self.db.pts_update_job(id=self.latest_id, status="error", job_dict=job_dict)
            print("error cases: {}".format(error_dict))
            raise Exception("something wrong with api benchmark PTS job id: {} !!".format(self.latest_id))
        else:
            self.db.pts_update_job(id=self.latest_id, status="done", job_dict=job_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yaml", type=str, help="input the yaml path")
    parser.add_argument("--id", type=int, default=0, help="job id")
    # parser.add_argument("--routine", type=int, default=1, help="if 1, daily routine mission")
    parser.add_argument("--comment", type=str, default=None, help="your comment")
    parser.add_argument("--framework", type=str, default="paddle", help="[paddle] | [torch]")
    parser.add_argument("--enable_backward", type=int, default=1, help="if 1, enable backward test")
    parser.add_argument(
        "--python", type=str, default="python3.8", help="python version like python3.7 | python3.8 etc."
    )
    parser.add_argument("--place", type=str, default="cpu", help="[cpu] or [gpu]")
    parser.add_argument("--yaml_info", type=str, default="case_0", help="[case_0] or [case_1] or [case_2]")
    parser.add_argument("--wheel_link", type=str, default=None, help="paddle wheel link")
    args = parser.parse_args()

    api_bm = ApiBenchmarkPTS(
        yaml_path=args.yaml,
        latest_id=args.id,
        comment=args.comment,
        framework=args.framework,
        enable_backward=args.enable_backward,
        python=args.python,
        place=args.place,
        yaml_info=args.yaml_info,
        wheel_link=args.wheel_link,
    )
    api_bm._run_pts()
