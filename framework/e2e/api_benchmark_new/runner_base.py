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

from statistics.statistics import Statistics

import paddle

sys.path.append("..")
from utils.yaml_loader import YamlLoader
from utils.logger import Logger
from benchtrans import BenchTrans
from jelly.jelly_v2 import Jelly_v2
from jelly.jelly_v2_torch import Jelly_v2_torch


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
        # 测试控制项
        self.storage = "apibm_config.yml"
        self.loops = 50  # 循环次数
        self.base_times = 1000  # timeit 基础运行时间
        self.default_dtype = "float32"

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
        if case_name in SKIP_DICT[platform.system()]:
            self.logger.get_log().warning("skip case -->{}<--".format(case_name))
            return
        if SPECIAL and case_name not in SKIP_DICT[platform.system()]:
            self.logger.get_log().warning("case is not in index_dict, skipping...-->{}<--".format(case_name))
            return
        if self.yaml_info == "case_0":
            if not case_name.endswith("_0"):
                self.logger.get_log().warning("skip case -->{}<--".format(case_name))
                return
        if self.yaml_info == "case_1":
            if case_name.endswith("_2"):
                self.logger.get_log().warning("skip case -->{}<--".format(case_name))
                return
        if self.yaml_info == "case_2":
            if not case_name.endswith("_2"):
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
            if self.framework == "torch":
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
