#!/bin/env python3
# -*- coding: utf-8 -*-
# @author Zeref996
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
runner
"""

import platform
import traceback
import sys
from datetime import datetime

from statistics.statistics import Statistics
from db.db import DB

import paddle

sys.path.append("..")
from utils.yaml_loader import YamlLoader
from utils.logger import Logger
from benchtrans import BenchTrans
from jelly.jelly_v2 import Jelly_v2

# from jelly.jelly_v2_torch import Jelly_v2_torch


SKIP_DICT = {"Windows": ["fft"], "Darwin": ["fft"], "Linux": []}
INDEX_DICT = {}
SPECIAL = False  # speacial for emergency


class ApiBenchmarkBASE(object):
    """
    api benchmark 调度CI, 监控cpu+前向, 支持多个机器baseline
    """

    def __init__(self, yaml_path):
        """
        :param baseline: 性能baseline键值对, key为case名, value为性能float
        """
        # apibm数据库信息
        self.storage = "apibm_config.yml"

        # 测试控制项
        self.loops = 50  # 循环次数
        self.base_times = 1000  # timeit 基础运行时间
        self.default_dtype = "float32"
        self.if_showtime = True
        self.double_check = True
        self.check_iters = 5
        self.now_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 初始化数据库
        self.db = DB(storage=self.storage)

        # 框架信息
        self.framework = "paddle"

        # 获取所有case名称
        self.yaml_path = yaml_path
        self.yaml_loader = YamlLoader(self.yaml_path)
        self.all_cases = self.yaml_loader.get_all_case_name()

        # 项目配置信息
        self.place = "cpu"
        # self.python = "python37"
        self.enable_backward = 0
        self.yaml_info = "case_0"
        self.card = 0

        # 初始化日志
        self.logger = Logger("ApiBenchmarkBASE")

        # 初始化统计模块
        self.statistics = Statistics()

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

    def _run_main(self, all_cases, latest_id):
        """
        对指定case运行测试
        :param all_cases: list of cases
        :param latest_id: 任务jid
        :param iters: 迭代次数
        :return:
        """
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

                self.db.insert_case(jid=latest_id, data_dict=latest_case, create_time=self.now_time)
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
