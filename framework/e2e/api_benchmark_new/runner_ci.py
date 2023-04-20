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
import copy

import argparse
import json
import sys
from datetime import datetime

from statistics.statistics import Statistics
from db.ci_db import CIdb
from info.snapshot import Snapshot
from strategy.compare import data_dict_compare, double_check, ci_level_reveal
from strategy.transdata import data_list_to_dict
from alarm.alarm import Alarm

import paddle

sys.path.append("..")
from utils.yaml_loader import YamlLoader
from utils.logger import Logger
from benchtrans import BenchTrans
from jelly.jelly_v2 import Jelly_v2
from tools import delete


SKIP_DICT = {"Windows": ["fft"], "Darwin": ["fft"], "Linux": []}
INDEX_DICT = {}
SPECIAL = False  # speacial for emergency


class ApiBenchmarkCI(object):
    """
    api benchmark 调度CI, 监控cpu+前向, 支持多个机器baseline
    """

    def __init__(self, yaml_path):
        """
        :param baseline: 性能baseline键值对, key为case名, value为性能float
        """
        # 测试控制项
        self.storage = "apibm_config.yml"
        self.loops = 50  # 循环次数
        self.base_times = 1000  # timeit 基础运行时间
        self.default_dtype = "float32"
        self.if_showtime = True
        self.double_check = True
        self.check_iters = 5

        # 初始化数据库
        self.db = CIdb(storage=self.storage)

        # md5唯一标识码
        self.md5_id = Snapshot().get_md5_id()

        # 例行标识
        self.baseline_comment = "baseline_CI_api_benchmark_pr_dev"
        # self.comment = "CI_api_benchmark_pr_dev"
        # self.baseline_comment = "naive_test"
        self.comment = "naive_test"
        self.routine = 0
        self.ci = 1
        self.uid = -1

        # 查询数据库构建baseline
        self.baseline_id = self.db.ci_select_baseline_job(
            comment=self.baseline_comment, routine=1, ci=self.ci, md5_id=self.md5_id
        )
        # self.baseline_id = 71
        self.baseline_list = self.db.select(table="case", condition_list=["jid = {}".format(self.baseline_id)])
        self.baseline_dict = data_list_to_dict(self.baseline_list)

        # 效率云环境变量
        self.AGILE_PULL_ID = os.environ.get("AGILE_PULL_ID", "0")
        self.AGILE_REVISION = os.environ.get("AGILE_REVISION", "0")
        self.AGILE_PIPELINE_BUILD_ID = os.environ.get("AGILE_PIPELINE_BUILD_ID", 0)
        self.description = {"success": True, "reason": "ok", "pipelineBuildId": self.AGILE_PIPELINE_BUILD_ID}

        # paddle信息
        self.framework = "paddle"
        self.wheel_link = (
            "https://xly-devops.bj.bcebos.com/PR/build_whl/{}/{}"
            "/paddlepaddle_gpu-0.0.0-cp37-cp37m-linux_x86_64.whl".format(self.AGILE_PULL_ID, self.AGILE_REVISION)
        )

        # paddle信息callback
        self.commit = paddle.__git_commit__
        self.version = paddle.__version__
        self.cuda = paddle.version.cuda()
        self.cudnn = paddle.version.cudnn()

        # 项目配置信息
        self.place = "cpu"
        self.python = "python37"
        self.enable_backward = 0
        self.yaml_info = "case_0"
        self.card = 0

        # 机器系统信息
        self.hostname = socket.gethostname()
        self.system = platform.system()
        self.snapshot = {
            "os": platform.platform(),
            "card": self.card,
            "cuda": paddle.version.cuda(),
            "cudnn": paddle.version.cudnn(),
            "comment": self.comment,
        }

        # 获取所有case名称
        self.yaml_path = yaml_path
        self.yaml_loader = YamlLoader(self.yaml_path)
        self.all_cases = self.yaml_loader.get_all_case_name()

        # 初始化日志
        self.logger = Logger("ApiBenchmarkCI")
        # Logger("RunnerCI").get_log()
        # exit(0)

        # 初始化统计模块
        self.statistics = Statistics()

        # 邮件报警
        # self.email = Alarm(storage=self.storage)

    def single_run(self, case_name):
        """
        运行单个case
        """
        if case_name in SKIP_DICT[platform.system()]:
            self.logger.get_log().warning("skip case -->{}<--".format(case_name))
            return
        if SPECIAL and case_name not in SKIP_DICT[platform.system()]:
            self.logger.get_log().warning("case is not in index_dict, skipping...-->{}<--".format(case_name))
            return
        if not case_name.endswith("_0"):
            self.logger.get_log().warning("skip case -->{}<--".format(case_name))
            return

        case_info = self.yaml_loader.get_case_info(case_name)
        bt = BenchTrans(case=case_info, logger=self.logger)
        if self.enable_backward == 0:
            enable_backward_trigger = False
        else:
            enable_backward_trigger = bt.enable_backward()
        api = bt.get_paddle_api()

        try:
            jelly = Jelly_v2(
                api=api,
                logger=self.logger,
                title=case_name,
                place=self.place,
                card=self.card,
                default_dtype=self.default_dtype,
                enable_backward=enable_backward_trigger,
            )
            jelly.set_paddle_param(bt.get_paddle_inputs(), bt.get_paddle_param())
            jelly.set_paddle_method(bt.get_paddle_method())

            if enable_backward_trigger:
                forward_time_list = jelly.paddle_forward()
                total_time_list = jelly.paddle_total()
                backward_time_list = list(map(lambda x: x[0] - x[1], zip(total_time_list, forward_time_list)))
            else:
                forward_time_list = jelly.paddle_forward()
                total_time_list = jelly.paddle_forward()
                backward_time_list = list(map(lambda x: x[0] - x[1], zip(total_time_list, forward_time_list)))

        except Exception as e:
            # 存储异常
            forward_time_list = traceback.format_exc()
            total_time_list = "error"
            backward_time_list = "error"
            paddle.enable_static()
            paddle.disable_static()
            self.logger.get_log().warning(e)

        return forward_time_list, total_time_list, backward_time_list, api

    def run(self, all_cases, latest_id, iters):
        """
        对指定case运行测试
        :param all_cases: list of cases
        :param latest_id: 任务jid
        :param iters: 迭代次数
        :return:
        """
        latest_dict = {}
        error_dict = {}

        for case_name in all_cases:
            tmp = {}
            error = {}

            forward = None
            total = None
            backward = None
            best_total = None

            forward_list = []
            total_list = []
            backward_list = []
            best_total_list = []

            forward_time_list, total_time_list, backward_time_list, api = self.single_run(case_name)

            if isinstance(forward_time_list, str):
                forward = "error"
                backward = "error"
                total = "error"
                best_total = "error"
                print("forward is: ", forward)
                print("backward is: ", backward)
                error["api"] = api
                error["exception"] = forward_time_list
                error_dict[case_name] = error
            elif isinstance(forward_time_list, list):
                forward_time_statistics = self.statistics.trimmean(data_list=forward_time_list)
                backward_time_statistics = self.statistics.trimmean(data_list=backward_time_list)
                total_time_statistics = self.statistics.trimmean(data_list=total_time_list)
                total_time_best = self.statistics.best(data_list=total_time_list)

                forward_list.append(forward_time_statistics)
                backward_list.append(backward_time_statistics)
                total_list.append(total_time_statistics)
                best_total_list.append(total_time_best)

                if self.if_showtime:
                    self._show(
                        forward_time=forward_time_statistics,
                        backward_time=backward_time_statistics,
                        total_time=total_time_statistics,
                        best_total_time=total_time_best,
                    )

                for i in range(iters - 1):
                    forward_time_list, total_time_list, backward_time_list, api = self.single_run(case_name)

                    forward_time_statistics = self.statistics.trimmean(data_list=forward_time_list)
                    backward_time_statistics = self.statistics.trimmean(data_list=backward_time_list)
                    total_time_statistics = self.statistics.trimmean(data_list=total_time_list)
                    total_time_best = self.statistics.best(data_list=total_time_list)

                    forward_list.append(forward_time_statistics)
                    backward_list.append(backward_time_statistics)
                    total_list.append(total_time_statistics)
                    best_total_list.append(total_time_best)
            else:
                raise Exception("when ApiBenchmarkCI.run(), something wrong with case: {}".format(case_name))

            if bool(forward_list) and bool(backward_list) and bool(total_list):
                forward = self.statistics.best(data_list=forward_list)
                backward = self.statistics.best(data_list=backward_list)
                total = self.statistics.best(data_list=total_list)
                best_total = self.statistics.best(data_list=best_total_list)

            tmp["jid"] = latest_id
            tmp["case_name"] = case_name
            tmp["api"] = api
            tmp["result"] = {
                "api": api,
                "yaml": case_name,
                "forward": forward,
                "total": total,
                "backward": backward,
                "best_total": best_total,
            }

            latest_dict[case_name] = tmp

        return latest_dict, error_dict

    def ci_test(self):
        """

        :return:
        """
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
            create_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            update_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

        latest_dict, error_dict = self.run(all_cases=self.all_cases, latest_id=latest_id, iters=1)

        # compare逻辑需要修改latest_list到latest_dict
        compare_res = data_dict_compare(baseline_data=self.baseline_dict, latest_data=latest_dict)

        # double_check_list = []
        if self.double_check:
            double_check_list = double_check(res=compare_res)
            latest_check_dict, error_check_dict = self.run(
                all_cases=double_check_list, latest_id=latest_id, iters=self.check_iters
            )
            check_res = data_dict_compare(baseline_data=self.baseline_dict, latest_data=latest_check_dict)

            # 使用double check的结果，更新原始数据
            for case_name, data_dict in latest_check_dict.items():
                latest_dict[case_name]["result"] = data_dict["result"]
                compare_res[case_name] = check_res[case_name]

        job_dict = copy.deepcopy(latest_dict)
        self.db.ci_update_job_and_insert_case(job_id=latest_id, job_dict=job_dict, error_dict=error_dict)

        # print('latest_dict is: ', latest_dict)
        # print('error_dict is: ', error_dict)
        # print('double_check_list is: ', double_check_list)
        # print('compare_res is: ', compare_res)

        api_grade = ci_level_reveal(compare_res)
        print(api_grade)

    def baseline_insert(self):
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
            wheel_link=self.wheel_link,
            description=json.dumps(self.description),
            create_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            update_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

        job_dict, error_dict = self.run(all_cases=self.all_cases, latest_id=job_id, iters=1)

        self.db.ci_update_job_and_insert_case(job_id=job_id, job_dict=job_dict, error_dict=error_dict)

    def _show(self, forward_time, backward_time, total_time, best_total_time):
        """
        logger 打印
        """
        self.logger.get_log().info("{} {} times forward cost {}s".format(self.framework, self.base_times, forward_time))
        self.logger.get_log().info(
            "{} {} times backward cost {}s".format(self.framework, self.base_times, backward_time)
        )
        self.logger.get_log().info("{} {} times total cost {}s".format(self.framework, self.base_times, total_time))
        self.logger.get_log().info(
            "{} {} times best_total cost {}s".format(self.framework, self.base_times, best_total_time)
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yaml", type=str, help="input the yaml path")
    args = parser.parse_args()

    api_bm = ApiBenchmarkCI(yaml_path="./../yaml/test0.yml")
    api_bm.ci_test()
    # api_bm.baseline_insert()
