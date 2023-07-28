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
from db.xly_db import XLYdb
from info.snapshot import Snapshot
from alarm.alarm import Alarm

sys.path.append("..")
from utils.yaml_loader import YamlLoader
from utils.logger import Logger
from benchtrans import BenchTrans
from jelly.jelly_v2 import Jelly_v2

import paddle

import psutil

p = psutil.Process()
p.cpu_affinity([2])

SKIP_DICT = {"Windows": ["fft"], "Darwin": ["fft"], "Linux": []}
INDEX_DICT = {}
SPECIAL = False  # speacial for emergency


class ApiBenchmarkNoDB(object):
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

        # 初始化数据库
        # self.db = XLYdb(storage=self.storage)

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

    def _run_test(self, case_name):
        """
        运行单个case
        """
        case_info = self.yaml_loader.get_case_info(case_name)
        bt = BenchTrans(case=case_info, logger=self.logger)
        if self.enable_backward == 0:
            enable_backward_trigger = False
        else:
            enable_backward_trigger = bt.enable_backward()
        api = bt.get_paddle_api()

        try:
            if self.framework == "torch":
                from jelly.jelly_v2_torch import Jelly_v2_torch

                jelly = Jelly_v2_torch(
                    api=api,
                    logger=self.logger,
                    title=case_name,
                    place=self.place,
                    card=self.card,
                    default_dtype=self.default_dtype,
                    loops=self.loops,
                    base_times=self.base_times,
                )
            else:
                jelly = Jelly_v2(
                    api=api,
                    logger=self.logger,
                    title=case_name,
                    place=self.place,
                    card=self.card,
                    default_dtype=self.default_dtype,
                    loops=self.loops,
                    base_times=self.base_times,
                )
            jelly.set_paddle_param(bt.get_paddle_inputs(), bt.get_paddle_param())
            jelly.set_paddle_method(bt.get_paddle_method())

            if enable_backward_trigger:
                forward_time_list = jelly.paddle_forward()
                total_time_list = jelly.paddle_total()
                backward_time_list = list(map(lambda x: x[0] - x[1], zip(total_time_list, forward_time_list)))
            else:
                forward_time_list = jelly.paddle_forward()
                total_time_list = forward_time_list
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

    def _run_main(self, all_cases):
        """
        对指定case运行测试
        :param all_cases: list of cases
        :param iters: 迭代次数
        :return:
        """
        error_dict = {}

        for case_name in all_cases:
            error = {}
            # latest_case = {}

            # forward_res_list = []
            # total_res_list = []
            # backward_res_list = []
            # best_total_res_list = []

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

            for k in range(self.experi):
                forward_res_list = []
                total_res_list = []
                backward_res_list = []
                best_total_res_list = []
                forward_time_list, total_time_list, backward_time_list, api = self._run_test(case_name)

                self.logger.get_log().info("experi {} start~~~~".format(k))
                self.logger.get_log().warning("forward_time_list is: {}".format(forward_time_list))
                self.logger.get_log().warning("forward_time_list length is: {}".format(len(forward_time_list)))
                self.logger.get_log().warning("total_time_list is: {}".format(total_time_list))
                self.logger.get_log().warning("total_time_list length is: {}".format(len(total_time_list)))

                if isinstance(forward_time_list, str):
                    error["api"] = api
                    error["exception"] = forward_time_list
                    error_dict[case_name] = error
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
                    # print('len(forward_time_list) is: ', len(forward_time_list))
                    #
                    # print('len(forward_res_list) is: ', len(forward_res_list))
                    # print('len(best_total_res_list) is: ', len(best_total_res_list))

                    forward = self.statistics.best(data_list=forward_res_list)
                    # backward = self.statistics.best(data_list=backward_res_list)
                    # total = self.statistics.best(data_list=total_res_list)
                    best_total = self.statistics.best(data_list=best_total_res_list)

                    self.logger.get_log().warning("forward_res_list is: {}".format(forward_res_list))
                    self.logger.get_log().warning("forward_res_list length is: {}".format(len(forward_res_list)))
                    self.logger.get_log().warning("total_res_list is: {}".format(total_res_list))
                    self.logger.get_log().warning("total_res_list length is: {}".format(len(total_res_list)))
                    self.logger.get_log().warning("best_total_res_list is: {}".format(best_total_res_list))
                    self.logger.get_log().warning("best_total_res_list length is: {}".format(len(best_total_res_list)))

                    # self.logger.get_log().info(
                    #     "experi {} forward_time_list: {}".format(
                    #         k, forward_time_list
                    #     )
                    # )
                    if self.if_showtime:
                        self._show(experi_idx=k, forward_time=forward, best_total_time=best_total)

                    # latest_case["jid"] = latest_id
                    # latest_case["case_name"] = case_name
                    # latest_case["api"] = api
                    # latest_case["result"] = {
                    #     "api": api,
                    #     "yaml": case_name,
                    #     "forward": forward,
                    #     "total": total,
                    #     "backward": backward,
                    #     "best_total": best_total,
                    # }

                    # self.db.insert_case(jid=latest_id, data_dict=latest_case, create_time=self.now_time)
                else:
                    raise Exception("when ApiBenchmark.run(), something wrong with case: {}".format(case_name))

        return error_dict

    def _base_statistics(
        self,
        forward_res_list,
        backward_res_list,
        total_res_list,
        best_total_res_list,
        forward_time_list,
        backward_time_list,
        total_time_list,
    ):
        """

        :return:
        """
        forward_time_statistics = self.statistics.trimmean(data_list=forward_time_list)
        backward_time_statistics = self.statistics.trimmean(data_list=backward_time_list)
        total_time_statistics = self.statistics.trimmean(data_list=total_time_list)
        total_time_best = self.statistics.best(data_list=total_time_list)

        forward_res_list.append(forward_time_statistics)
        backward_res_list.append(backward_time_statistics)
        total_res_list.append(total_time_statistics)
        best_total_res_list.append(total_time_best)

        return forward_res_list, backward_res_list, total_res_list, best_total_res_list

    def _show(self, experi_idx, forward_time, best_total_time):
        """
        logger 打印
        """
        self.logger.get_log().info(
            "experi {} result: forward cost {}s, best_forward cost {}s".format(
                experi_idx, forward_time, best_total_time
            )
        )

    def _run_xly(self):
        """

        :return:
        """
        # latest_id = self.db.xly_insert_job(
        #     framework=self.framework,
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
        #     routine=self.routine,
        #     # ci=self.ci,
        #     comment=self.comment,
        #     enable_backward=self.enable_backward,
        #     python=self.python,
        #     yaml_info=self.yaml_info,
        #     wheel_link=self.wheel_link,
        #     description=json.dumps(self.description),
        #     create_time=self.now_time,
        #     update_time=self.now_time,
        # )

        error_dict = self._run_main(all_cases=self.all_cases)
        del error_dict

        # if bool(error_dict):
        #     self.db.xly_update_job(id=latest_id, status="error", update_time=self.now_time)
        #     print("error cases: {}".format(error_dict))
        #     raise Exception("something wrong with api benchmark XLY job id: {} !!".format(latest_id))
        # else:
        #     self.db.xly_update_job(id=latest_id, status="done", update_time=self.now_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    # yaml_path, routine, comment, framework, enable_backward, python, place, yaml_info, wheel_link
    parser.add_argument("--yaml", type=str, help="input the yaml path")
    parser.add_argument("--framework", type=str, default="paddle", help="[paddle] | [torch]")
    parser.add_argument("--enable_backward", type=int, default=1, help="if 1, enable backward test")
    parser.add_argument("--place", type=str, default="cpu", help="[cpu] or [gpu]")
    parser.add_argument("--yaml_info", type=str, default="case_0", help="[case_0] or [case_1] or [case_2]")
    args = parser.parse_args()

    api_bm = ApiBenchmarkNoDB(
        yaml_path=args.yaml,
        framework=args.framework,
        enable_backward=args.enable_backward,
        place=args.place,
        yaml_info=args.yaml_info,
    )
    api_bm._run_xly()
    # python runner_nodb.py --yaml broadcast_shape.yml --framework paddle
    # --enable_backward 0 --place cpu 2>&1 | tee -a f_broadcast_1w.log
