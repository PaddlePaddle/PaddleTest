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
from db.xly_db import XLYdb
from info.snapshot import Snapshot
from alarm.alarm import Alarm

import paddle

sys.path.append("..")
from utils.logger import Logger
from runner_base import ApiBenchmarkBASE


SKIP_DICT = {"Windows": ["fft"], "Darwin": ["fft"], "Linux": []}
INDEX_DICT = {}
SPECIAL = False  # speacial for emergency


class ApiBenchmarkXLY(ApiBenchmarkBASE):
    """
    api benchmark 调度CI, 监控cpu+前向, 支持多个机器baseline
    """

    def __init__(self, yaml_path, routine, comment, framework, enable_backward, python, place, yaml_info, wheel_link):
        super(ApiBenchmarkXLY, self).__init__(yaml_path)
        """
        :param baseline: 性能baseline键值对, key为case名, value为性能float
        """
        # 测试控制项
        self.loops = 1000  # 循环次数
        self.base_times = 50  # timeit 基础运行时间
        self.default_dtype = "float32"
        self.if_showtime = True
        self.double_check = True
        self.check_iters = 5
        self.now_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 初始化数据库
        self.db = XLYdb(storage=self.storage)

        # md5唯一标识码
        self.md5_id = Snapshot().get_md5_id()

        # 效率云环境变量
        self.AGILE_PIPELINE_BUILD_ID = os.environ.get("AGILE_PIPELINE_BUILD_ID", 0)
        self.description = {"success": True, "reason": "ok", "pipelineBuildId": self.AGILE_PIPELINE_BUILD_ID}

        # 例行标识
        # self.latest_id = latest_id
        self.comment = comment
        self.routine = routine
        self.ci = 0
        self.uid = -1

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

    # def _run_main(self, all_cases, latest_id):
    #     """
    #     对指定case运行测试
    #     :param all_cases: list of cases
    #     :param latest_id: 任务jid
    #     :param iters: 迭代次数
    #     :return:
    #     """
    #     error_dict = {}
    #
    #     for case_name in all_cases:
    #         error = {}
    #         latest_case = {}
    #
    #         forward_res_list = []
    #         total_res_list = []
    #         backward_res_list = []
    #         best_total_res_list = []
    #
    #         if case_name in SKIP_DICT[platform.system()]:
    #             self.logger.get_log().warning("skip case -->{}<--".format(case_name))
    #             continue
    #         if SPECIAL and case_name not in SKIP_DICT[platform.system()]:
    #             self.logger.get_log().warning("case is not in index_dict, skipping...-->{}<--".format(case_name))
    #             continue
    #         if self.yaml_info == "case_0":
    #             if not case_name.endswith("_0"):
    #                 self.logger.get_log().warning("skip case -->{}<--".format(case_name))
    #                 continue
    #         if self.yaml_info == "case_1":
    #             if case_name.endswith("_2"):
    #                 self.logger.get_log().warning("skip case -->{}<--".format(case_name))
    #                 continue
    #         if self.yaml_info == "case_2":
    #             if not case_name.endswith("_2"):
    #                 self.logger.get_log().warning("skip case -->{}<--".format(case_name))
    #                 continue
    #
    #         forward_time_list, total_time_list, backward_time_list, api = self._run_test(case_name)
    #
    #         if isinstance(forward_time_list, str):
    #             error["api"] = api
    #             error["exception"] = forward_time_list
    #             error_dict[case_name] = error
    #         elif isinstance(forward_time_list, list):
    #             forward_res_list, backward_res_list, total_res_list, best_total_res_list = self._base_statistics(
    #                 forward_res_list=forward_res_list,
    #                 backward_res_list=backward_res_list,
    #                 total_res_list=total_res_list,
    #                 best_total_res_list=best_total_res_list,
    #                 forward_time_list=forward_time_list,
    #                 backward_time_list=backward_time_list,
    #                 total_time_list=total_time_list,
    #             )
    #
    #             forward = self.statistics.best(data_list=forward_res_list)
    #             backward = self.statistics.best(data_list=backward_res_list)
    #             total = self.statistics.best(data_list=total_res_list)
    #             best_total = self.statistics.best(data_list=best_total_res_list)
    #
    #             if self.if_showtime:
    #                 self._show(
    #                     forward_time=forward,
    #                     backward_time=backward,
    #                     total_time=total,
    #                     best_total_time=best_total,
    #                 )
    #
    #             latest_case["jid"] = latest_id
    #             latest_case["case_name"] = case_name
    #             latest_case["api"] = api
    #             latest_case["result"] = {
    #                 "api": api,
    #                 "yaml": case_name,
    #                 "forward": forward,
    #                 "total": total,
    #                 "backward": backward,
    #                 "best_total": best_total,
    #             }
    #
    #             self.db.insert_case(jid=latest_id, data_dict=latest_case, create_time=self.now_time)
    #         else:
    #             raise Exception("when ApiBenchmarkPTS.run(), something wrong with case: {}".format(case_name))
    #
    #     return error_dict

    def _run_pts(self):
        """

        :return:
        """
        latest_id = self.db.xly_insert_job(
            framework=self.framework,
            commit=self.commit,
            version=self.version,
            hostname=self.hostname,
            place=self.place,
            system=self.system,
            cuda=self.cuda,
            cudnn=self.cudnn,
            snapshot=json.dumps(self.snapshot),
            md5_id=self.md5_id,
            uid=self.uid,
            routine=self.routine,
            # ci=self.ci,
            comment=self.comment,
            enable_backward=self.enable_backward,
            python=self.python,
            yaml_info=self.yaml_info,
            wheel_link=self.wheel_link,
            description=json.dumps(self.description),
            create_time=self.now_time,
            update_time=self.now_time,
        )

        error_dict = self._run_main(all_cases=self.all_cases, latest_id=latest_id)

        if bool(error_dict):
            self.db.xly_update_job(id=latest_id, status="error", update_time=self.now_time)
            print("error cases: {}".format(error_dict))
            raise Exception("something wrong with api benchmark XLY job id: {} !!".format(latest_id))
        else:
            self.db.xly_update_job(id=latest_id, status="done", update_time=self.now_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yaml", type=str, help="input the yaml path")
    args = parser.parse_args()

    api_bm = ApiBenchmarkXLY(
        yaml_path="./../yaml/test0.yml",
        # latest_id=41,
        routine=0,
        comment="fake_run_xly",
        framework="paddle",
        enable_backward=1,
        python="python38",
        place="cpu",
        yaml_info="case_0",
        wheel_link="fake_link",
    )
    api_bm._run_pts()
    # api_bm.baseline_insert()
