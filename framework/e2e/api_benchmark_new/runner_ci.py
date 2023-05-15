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
from db.ci_db import CIdb
from info.snapshot import Snapshot
from strategy.compare import double_check, ci_level_reveal, data_compare
from strategy.transdata import data_list_to_dict
from alarm.alarm import Alarm

import paddle

sys.path.append("..")
from utils.logger import Logger
from runner_base import ApiBenchmarkBASE


SKIP_DICT = {"Windows": ["fft"], "Darwin": ["fft"], "Linux": []}
INDEX_DICT = {}
SPECIAL = False  # speacial for emergency


class ApiBenchmarkCI(ApiBenchmarkBASE):
    """
    api benchmark 调度CI, 监控cpu+前向, 支持多个机器baseline
    """

    def __init__(self, yaml_path):
        super(ApiBenchmarkCI, self).__init__(yaml_path)
        """
        :param baseline: 性能baseline键值对, key为case名, value为性能float
        """
        # 测试控制项
        self.loops = 50  # 循环次数
        self.base_times = 1000  # timeit 基础运行时间
        self.default_dtype = "float32"
        self.if_showtime = True
        self.double_check = True
        self.check_iters = 5
        self.now_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 初始化数据库
        self.db = CIdb(storage=self.storage)

        # md5唯一标识码
        self.md5_id = Snapshot().get_md5_id()

        # 效率云环境变量
        self.AGILE_PULL_ID = os.environ.get("AGILE_PULL_ID", "0")
        self.AGILE_REVISION = os.environ.get("AGILE_REVISION", "0")
        self.AGILE_PIPELINE_BUILD_ID = os.environ.get("AGILE_PIPELINE_BUILD_ID", 0)
        self.description = {"success": True, "reason": "ok", "pipelineBuildId": self.AGILE_PIPELINE_BUILD_ID}

        # 例行标识
        self.baseline_comment = "baseline_CI_api_benchmark_pr_dev"
        self.comment = "CI_api_benchmark_pr_{}_ver_{}".format(self.AGILE_PULL_ID, self.AGILE_REVISION)
        self.routine = 0
        self.ci = 1
        self.uid = -1

        # # 查询数据库构建baseline
        # self.baseline_id = self.db.ci_select_baseline_job(
        #     comment=self.baseline_comment, routine=1, ci=self.ci, md5_id=self.md5_id
        # )
        # # self.baseline_id = 123
        # self.baseline_list = self.db.select(table="case", condition_list=["jid = {}".format(self.baseline_id)])
        # self.baseline_dict = data_list_to_dict(self.baseline_list)

        # 框架信息
        self.framework = "paddle"
        self.wheel_link = (
            "https://xly-devops.bj.bcebos.com/PR/build_whl/{}/{}"
            "/paddlepaddle_gpu-0.0.0-cp37-cp37m-linux_x86_64.whl".format(self.AGILE_PULL_ID, self.AGILE_REVISION)
        )

        # 框架信息callback
        self.commit = paddle.__git_commit__
        self.version = paddle.__version__
        self.cuda = paddle.version.cuda()
        self.cudnn = paddle.version.cudnn()

        # 项目配置信息
        self.place = "cpu"
        self.python = "python3.7"
        self.enable_backward = 0
        self.yaml_info = "case_0"
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
        self.logger = Logger("ApiBenchmarkCI")

        # 初始化统计模块
        self.statistics = Statistics()

        # 邮件报警
        # self.email = Alarm(storage=self.storage)

    def _run_main(self, all_cases, latest_id, iters, compare_switch):
        """
        对指定case运行测试
        :param all_cases: list of cases
        :param latest_id: 任务jid
        :param iters: 迭代次数
        :return:
        """
        compare_dict = {}
        error_dict = {}

        for case_name in all_cases:
            error = {}
            latest_case = {}

            forward_res_list = []
            total_res_list = []
            backward_res_list = []
            best_total_res_list = []

            if case_name in SKIP_DICT[platform.system()]:
                self.logger.get_log().warning("skip case -->{}<--".format(case_name))
                continue
            if SPECIAL and case_name not in SKIP_DICT[platform.system()]:
                self.logger.get_log().warning("case is not in index_dict, skipping...-->{}<--".format(case_name))
                continue
            if self.yaml_info == "case_0":
                if not case_name.endswith("_0"):
                    self.logger.get_log().warning("skip case -->{}<--".format(case_name))
                    continue
            if self.yaml_info == "case_1":
                if case_name.endswith("_2"):
                    self.logger.get_log().warning("skip case -->{}<--".format(case_name))
                    continue
            if self.yaml_info == "case_2":
                if not case_name.endswith("_2"):
                    self.logger.get_log().warning("skip case -->{}<--".format(case_name))
                    continue

            forward_time_list, total_time_list, backward_time_list, api = self._run_test(case_name)

            if isinstance(forward_time_list, str):
                forward = "error"
                backward = "error"
                total = "error"
                best_total = "error"
                error["api"] = api
                error["exception"] = forward_time_list
                error_dict[case_name] = error

                if self.if_showtime:
                    self._show(
                        forward_time=forward,
                        backward_time=backward,
                        total_time=total,
                        best_total_time=best_total,
                    )

                if compare_switch:
                    compare_dict[case_name] = {
                        "baseline_api": self.baseline_dict[case_name]["api"],
                        "latest_api": api,
                        "forward": forward,
                        "backward": backward,
                        "total": total,
                    }

            elif isinstance(forward_time_list, list):
                forward_res_list, backward_res_list, total_res_list, best_total_res_list = self._base_statistics(
                    forward_res_list=forward_res_list,
                    backward_res_list=backward_res_list,
                    total_res_list=total_res_list,
                    best_total_res_list=best_total_res_list,
                    forward_time_list=forward_time_list,
                    backward_time_list=backward_time_list,
                    total_time_list=total_time_list,
                )

                forward = self.statistics.best(data_list=forward_res_list)
                backward = self.statistics.best(data_list=backward_res_list)
                total = self.statistics.best(data_list=total_res_list)
                best_total = self.statistics.best(data_list=best_total_res_list)

                if self.if_showtime:
                    self._show(
                        forward_time=forward,
                        backward_time=backward,
                        total_time=total,
                        best_total_time=best_total,
                    )

                latest_case["jid"] = latest_id
                latest_case["case_name"] = case_name
                latest_case["api"] = api
                latest_case["result"] = {
                    "api": api,
                    "yaml": case_name,
                    "forward": forward,
                    "total": total,
                    "backward": backward,
                    "best_total": best_total,
                }

                if compare_switch:
                    compare_res = data_compare(
                        baseline_case=self.baseline_dict[case_name], latest_case=latest_case, case_name=case_name
                    )
                    compare_dict[case_name] = compare_res[case_name]

                    if self.double_check and double_check(res=compare_res[case_name]):
                        for i in range(iters - 1):
                            forward_time_list, total_time_list, backward_time_list, api = self._run_test(case_name)

                            (
                                forward_res_list,
                                backward_res_list,
                                total_res_list,
                                best_total_res_list,
                            ) = self._base_statistics(
                                forward_res_list=forward_res_list,
                                backward_res_list=backward_res_list,
                                total_res_list=total_res_list,
                                best_total_res_list=best_total_res_list,
                                forward_time_list=forward_time_list,
                                backward_time_list=backward_time_list,
                                total_time_list=total_time_list,
                            )

                        # if bool(forward_res_list) and bool(backward_res_list) and bool(total_res_list):
                        forward = self.statistics.best(data_list=forward_res_list)
                        backward = self.statistics.best(data_list=backward_res_list)
                        total = self.statistics.best(data_list=total_res_list)
                        best_total = self.statistics.best(data_list=best_total_res_list)

                        latest_case["jid"] = latest_id
                        latest_case["case_name"] = case_name
                        latest_case["api"] = api
                        latest_case["result"] = {
                            "api": api,
                            "yaml": case_name,
                            "forward": forward,
                            "total": total,
                            "backward": backward,
                            "best_total": best_total,
                        }
                        compare_res = data_compare(
                            baseline_case=self.baseline_dict[case_name], latest_case=latest_case, case_name=case_name
                        )
                        compare_dict[case_name] = compare_res[case_name]

                self.db.insert_case(jid=latest_id, data_dict=latest_case, create_time=self.now_time)
            else:
                raise Exception("when ApiBenchmarkCI.run(), something wrong with case: {}".format(case_name))

        return compare_dict, error_dict

    def _run_ci(self):
        """

        :return:
        """
        # 查询数据库构建baseline
        self.baseline_id = self.db.ci_select_baseline_job(
            comment=self.baseline_comment, routine=1, ci=self.ci, md5_id=self.md5_id
        )
        # self.baseline_id = 123
        self.baseline_list = self.db.select(table="case", condition_list=["jid = {}".format(self.baseline_id)])
        self.baseline_dict = data_list_to_dict(self.baseline_list)

        latest_id = self.db.ci_insert_job(
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
            ci=self.ci,
            comment=self.comment,
            enable_backward=self.enable_backward,
            python=self.python,
            yaml_info=self.yaml_info,
            wheel_link=self.wheel_link,
            description=json.dumps(self.description),
            create_time=self.now_time,
            update_time=self.now_time,
        )

        compare_dict, error_dict = self._run_main(
            all_cases=self.all_cases, latest_id=latest_id, iters=self.check_iters, compare_switch=True
        )

        if bool(error_dict):
            self.db.ci_update_job(id=latest_id, status="error", update_time=self.now_time)
            raise Exception("something wrong with api benchmark CI job id: {} !!".format(latest_id))
        else:
            self.db.ci_update_job(id=latest_id, status="done", update_time=self.now_time)

        # print("compare_dict is: ", compare_dict)
        api_grade = ci_level_reveal(compare_dict)
        print(api_grade)

    def _baseline_insert(self, wheel_link):
        """

        :return:
        """
        job_id = self.db.ci_insert_job(
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
            routine=1,  # 基线例行标签
            ci=self.ci,
            comment=self.baseline_comment,  # 基线comment
            enable_backward=self.enable_backward,
            python=self.python,
            yaml_info=self.yaml_info,
            # wheel_link=self.wheel_link,
            wheel_link=wheel_link,
            description=json.dumps(self.description),
            create_time=self.now_time,
            update_time=self.now_time,
        )

        cases_dict, error_dict = self._run_main(
            all_cases=self.all_cases, latest_id=job_id, iters=1, compare_switch=False
        )

        if bool(error_dict):
            self.db.ci_update_job(id=job_id, status="error", update_time=self.now_time)
            raise Exception("something wrong with api benchmark job id: {} !!".format(job_id))
        else:
            self.db.ci_update_job(id=job_id, status="done", update_time=self.now_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yaml", type=str, help="input the yaml path")
    parser.add_argument("--baseline_whl_link", type=str, default=None, help="only be used to insert baseline data")
    args = parser.parse_args()

    # api_bm = ApiBenchmarkCI(yaml_path="./../yaml/api_benchmark_fp32.yml")
    api_bm = ApiBenchmarkCI(yaml_path=args.yaml)
    if bool(args.baseline_whl_link):
        api_bm._baseline_insert(wheel_link=args.baseline_whl_link)
    else:
        api_bm._run_ci()
