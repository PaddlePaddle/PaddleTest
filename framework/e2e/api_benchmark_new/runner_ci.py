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

# from db.db import DB
from db.ci_db import CIdb
from info.snapshot import Snapshot
from strategy.compare import double_check, ci_level_reveal, data_compare
from strategy.transdata import data_list_to_dict
from alarm.alarm import Alarm

import paddle

sys.path.append("..")
from utils.logger import Logger
from runner_base import ApiBenchmarkBASE

import psutil

p = psutil.Process()
p.cpu_affinity([2])

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

        # # # 初始化数据库
        # # self.db = CIdb(storage=self.storage)
        #
        # # md5唯一标识码
        # self.md5_id = Snapshot().get_md5_id()
        #
        # # 效率云环境变量
        # self.AGILE_PULL_ID = os.environ.get("AGILE_PULL_ID", "0")
        # self.AGILE_REVISION = os.environ.get("AGILE_REVISION", "0")
        # self.AGILE_PIPELINE_BUILD_ID = os.environ.get("AGILE_PIPELINE_BUILD_ID", 0)
        # self.description = {"success": True, "reason": "ok", "pipelineBuildId": self.AGILE_PIPELINE_BUILD_ID}
        #
        # # 例行标识
        # self.baseline_comment = "baseline_CI_api_benchmark_pr_dev"
        # self.comment = "CI_api_benchmark_pr_{}_ver_{}".format(self.AGILE_PULL_ID, self.AGILE_REVISION)
        # self.routine = 0
        # self.ci = 1
        # self.uid = -1
        #
        # # # 查询数据库构建baseline
        # # baseline_id = self.db.ci_select_baseline_job(
        # #     comment=self.baseline_comment, routine=1, ci=self.ci, md5_id=self.md5_id
        # # )
        # # # baseline_id = 123
        # # baseline_list = self.db.select(table="case", condition_list=["jid = {}".format(baseline_id)])
        # # baseline_dict = data_list_to_dict(baseline_list)
        #
        # # 框架信息
        # self.framework = "paddle"
        # self.wheel_link = (
        #     "https://xly-devops.bj.bcebos.com/PR/build_whl/{}/{}"
        #     "/paddlepaddle_gpu-0.0.0-cp37-cp37m-linux_x86_64.whl".format(self.AGILE_PULL_ID, self.AGILE_REVISION)
        # )
        #
        # # 框架信息callback
        # self.commit = paddle.__git_commit__
        # self.version = paddle.__version__
        # self.cuda = paddle.version.cuda()
        # self.cudnn = paddle.version.cudnn()
        #
        # # 项目配置信息
        # self.place = "cpu"
        # self.python = "python3.7"
        # self.enable_backward = 0
        # self.yaml_info = "case_0"
        # self.card = 0
        #
        # # 机器系统信息
        # self.hostname = socket.gethostname()
        # self.system = platform.system()
        # self.snapshot = {
        #     "os": platform.platform(),
        #     "card": self.card,
        #     "cuda": self.cuda,
        #     "cudnn": self.cudnn,
        #     "comment": self.comment,
        # }
        #
        # # 初始化日志
        # self.logger = Logger("ApiBenchmarkCI")
        #
        # # 初始化统计模块
        # self.statistics = Statistics()
        #
        # # 邮件报警
        # # self.email = Alarm(storage=self.storage)

    # def _run_ci(self):
    #     """
    #
    #     :return:
    #     """
    #     error_dict = self._run_main(all_cases=self.all_cases)
    #
    #     # 查询数据库构建baseline
    #     db = CIdb(storage=self.storage)
    #     baseline_id = db.ci_select_baseline_job(
    #         comment=self.baseline_comment, routine=1, ci=self.ci, md5_id=self.md5_id
    #     )
    #     # baseline_id = 123
    #     baseline_list = db.select(table="case", condition_list=["jid = {}".format(baseline_id)])
    #     baseline_dict = data_list_to_dict(baseline_list)  # 需要在_run_main之后添加对比逻辑，才会用到baseline_dict
    #     print(baseline_dict)
    #
    #     latest_id = db.ci_insert_job(
    #         commit=self.commit,
    #         version=self.version,
    #         hostname=self.hostname,
    #         place=self.place,
    #         system=self.system,
    #         cuda=self.cuda,
    #         cudnn=self.cudnn,
    #         snapshot=json.dumps(self.snapshot),
    #         md5_id=self.md5_id,
    #         uid=self.uid,
    #         routine=self.routine,
    #         ci=self.ci,
    #         comment=self.comment,
    #         enable_backward=self.enable_backward,
    #         python=self.python,
    #         yaml_info=self.yaml_info,
    #         wheel_link=self.wheel_link,
    #         description=json.dumps(self.description),
    #         create_time=self.now_time,
    #         update_time=self.now_time,
    #     )
    #
    #     # compare_dict, error_dict = self._run_main_ci(
    #     #     all_cases=self.all_cases, latest_id=latest_id, iters=self.check_iters, compare_switch=True
    #     # )
    #
    #     if bool(error_dict):
    #         db.ci_update_job(id=latest_id, status="error", update_time=self.now_time)
    #         raise Exception("something wrong with api benchmark CI job id: {} !!".format(latest_id))
    #     else:
    #         db.ci_update_job(id=latest_id, status="done", update_time=self.now_time)
    #
    #     # api_grade = ci_level_reveal(compare_dict)
    #     # del api_grade["equal"]
    #     # del api_grade["better"]
    #     print(
    #         "以下为pr{}引入之后，api调度性能相对于baseline的变化。worse表示性能下降超过30%的api，doubt表示性能下降为15%~30%之间的api".format(
    #             self.AGILE_PULL_ID
    #         )
    #     )
    #     # print(api_grade)
    #     print(
    #         "详情差异请点击以下链接查询: http://paddletest.baidu-int.com:8081/#/paddle/benchmark/apiBenchmark/report/{}&{}".format(
    #             latest_id, baseline_id
    #         )
    #     )
    #
    #     # if bool(api_grade["doubt"]) or bool(api_grade["worse"]):
    #     #     raise Exception("该pr会导致动态图cpu前向调度性能下降，请修复！！！")

    def _baseline_insert(self, wheel_link):
        """

        :return:
        """
        error_dict = self._run_main(all_cases=self.all_cases)
        # # 初始化数据库
        # db = CIdb(storage=self.storage)
        #
        # job_id = db.ci_insert_job(
        #     commit=self.commit,
        #     version=self.version,
        #     hostname=self.hostname,
        #     place=self.place,
        #     system=self.system,
        #     cuda=self.cuda,
        #     cudnn=self.cudnn,
        #     snapshot=json.dumps(self.snapshot),
        #     md5_id=self.md5_id,
        #     uid=self.uid,
        #     routine=1,  # 基线例行标签
        #     ci=self.ci,
        #     comment=self.baseline_comment,  # 基线comment
        #     enable_backward=self.enable_backward,
        #     python=self.python,
        #     yaml_info=self.yaml_info,
        #     # wheel_link=self.wheel_link,
        #     wheel_link=wheel_link,
        #     description=json.dumps(self.description),
        #     create_time=self.now_time,
        #     update_time=self.now_time,
        # )
        # # cases_dict, error_dict = self._run_main(
        # #     all_cases=self.all_cases, latest_id=job_id, iters=1, compare_switch=False
        # # )
        #
        # self._db_save(db=db, latest_id=job_id)
        #
        # if bool(error_dict):
        #     db.ci_update_job(id=job_id, status="error", update_time=self.now_time)
        #     print("error cases: {}".format(error_dict))
        #     raise Exception("something wrong with api benchmark job id: {} !!".format(job_id))
        # else:
        #     db.ci_update_job(id=job_id, status="done", update_time=self.now_time)
        #
        # # # debug
        # # cases_dict = self._run_main(all_cases=self.all_cases, latest_id=job_id)
        # # self.db.ci_update_job(id=job_id, status="done", update_time=self.now_time)
        # # del cases_dict
        print(error_dict)


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
