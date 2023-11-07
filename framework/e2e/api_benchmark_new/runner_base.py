#!/bin/env python3
# -*- coding: utf-8 -*-
# @author Zeref996
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
runner
"""
import os
import json
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

ACCURACY = "%.6g"

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

        # # 初始化数据库
        # self.db = DB(storage=self.storage)

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

    def _run_test(self, case_name, loops, base_times, log="log"):
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
                    loops=loops,
                    base_times=base_times,
                )
            else:
                jelly = Jelly_v2(
                    api=api,
                    logger=self.logger,
                    title=case_name,
                    place=self.place,
                    card=self.card,
                    default_dtype=self.default_dtype,
                    loops=loops,
                    base_times=base_times,
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
            forward = self.statistics.trimmean(data_list=forward_time_list, ratio=0.2)
            forward_top_k = self.statistics.best_top_k(data_list=forward_time_list, ratio=0.2)
            backward = self.statistics.trimmean(data_list=backward_time_list, ratio=0.2)
            total = self.statistics.trimmean(data_list=total_time_list, ratio=0.2)
            best_total = self.statistics.best(data_list=forward_time_list)

            jelly.result["forward"] = ACCURACY % forward
            jelly.result["forward_top_k"] = ACCURACY % forward_top_k
            jelly.result["backward"] = ACCURACY % backward
            jelly.result["total"] = ACCURACY % total
            jelly.result["best_total"] = ACCURACY % best_total

            self._log_save(data=jelly.result, case_name=case_name, log=log)

            self._show(
                forward_time=ACCURACY % forward,
                backward_time=ACCURACY % backward,
                total_time=ACCURACY % total,
                best_total_time=ACCURACY % best_total,
            )
            error_logo = False
            error_info = ""

        except Exception as e:
            # 存储异常
            error_info = traceback.format_exc()
            error_logo = True
            paddle.enable_static()
            paddle.disable_static()
            self.logger.get_log().warning(e)

        return error_logo, error_info, api

    def _run_main(self, all_cases, loops, base_times, log="log"):
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
            # latest_case = {}
            #
            # forward_res_list = []
            # forward_top_k_res_list = []
            # total_res_list = []
            # total_top_k_res_list = []
            # backward_res_list = []
            # backward_top_k_res_list = []
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

            error_logo, error_info, api = self._run_test(
                case_name=case_name, loops=loops, base_times=base_times, log=log
            )

            if error_logo:
                error["api"] = api
                error["exception"] = error_info
                error_dict[case_name] = error
            # elif error_logo:
            #     (
            #         forward_res_list,
            #         forward_top_k_res_list,
            #         backward_res_list,
            #         backward_top_k_res_list,
            #         total_res_list,
            #         total_top_k_res_list,
            #         best_total_res_list,
            #     ) = self._base_statistics(
            #         forward_res_list=forward_res_list,
            #         forward_top_k_res_list=forward_top_k_res_list,
            #         backward_res_list=backward_res_list,
            #         backward_top_k_res_list=backward_top_k_res_list,
            #         total_res_list=total_res_list,
            #         total_top_k_res_list=total_top_k_res_list,
            #         best_total_res_list=best_total_res_list,
            #         forward_time_list=forward_time_list,
            #         backward_time_list=backward_time_list,
            #         total_time_list=total_time_list,
            #     )
            #
            #     forward = self.statistics.best(data_list=forward_res_list)
            #     forward_top_k = self.statistics.best(data_list=forward_top_k_res_list)
            #
            #     backward = self.statistics.best(data_list=backward_res_list)
            #     # backward_top_k = self.statistics.best(data_list=backward_top_k_res_list)
            #
            #     total = self.statistics.best(data_list=total_res_list)
            #     # total_top_k = self.statistics.best(data_list=total_top_k_res_list)
            #
            #     best_total = self.statistics.best(data_list=best_total_res_list)
            #
            #     if self.if_showtime:
            #         self._show(
            #             forward_time=forward,
            #             backward_time=backward,
            #             total_time=total,
            #             best_total_time=best_total,
            #         )
            #
            #     latest_case["jid"] = latest_id
            #     latest_case["case_name"] = case_name
            #     latest_case["api"] = api
            #     latest_case["result"] = {
            #         "api": api,
            #         "yaml": case_name,
            #         "forward": forward,
            #         "forward_top_k": forward_top_k,
            #         "total": total,
            #         # "total_top_k": total_top_k,
            #         "backward": backward,
            #         # "backward_top_k": backward_top_k,
            #         "best_total": best_total,
            #     }
            #
            #     self.db.insert_case(jid=latest_id, data_dict=latest_case, create_time=self.now_time)
            # else:
            #     raise Exception("when ApiBenchmark.run(), something wrong with case: {}".format(case_name))

        return error_dict

    # def _base_statistics(
    #     self,
    #     forward_res_list,
    #     forward_top_k_res_list,
    #     backward_res_list,
    #     backward_top_k_res_list,
    #     total_res_list,
    #     total_top_k_res_list,
    #     best_total_res_list,
    #     forward_time_list,
    #     backward_time_list,
    #     total_time_list,
    # ):
    #     """
    #
    #     :return:
    #     """
    #     forward_time_statistics = self.statistics.trimmean(data_list=forward_time_list)
    #     forward_top_k_statistics = self.statistics.best_top_k(data_list=forward_time_list)
    #
    #     backward_time_statistics = self.statistics.trimmean(data_list=backward_time_list)
    #     backward_top_k_statistics = self.statistics.best_top_k(data_list=backward_time_list)
    #
    #     total_time_statistics = self.statistics.trimmean(data_list=total_time_list)
    #     total_top_k_statistics = self.statistics.best_top_k(data_list=total_time_list)
    #
    #     total_time_best = self.statistics.best(data_list=total_time_list)
    #
    #     forward_res_list.append(forward_time_statistics)
    #     forward_top_k_res_list.append(forward_top_k_statistics)
    #
    #     backward_res_list.append(backward_time_statistics)
    #     backward_top_k_res_list.append(backward_top_k_statistics)
    #
    #     total_res_list.append(total_time_statistics)
    #     total_top_k_res_list.append(total_top_k_statistics)
    #
    #     best_total_res_list.append(total_time_best)
    #
    #     return (
    #         forward_res_list,
    #         forward_top_k_res_list,
    #         backward_res_list,
    #         backward_top_k_res_list,
    #         total_res_list,
    #         total_top_k_res_list,
    #         best_total_res_list,
    #     )

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

    def _log_save(self, data, case_name, log="log"):
        """
        保存数据到磁盘
        :return:
        """
        log_file = "./{}/{}.json".format(log, case_name)
        if not os.path.exists("./{}".format(log)):
            os.makedirs("./{}".format(log))
        try:
            with open(log_file, "w") as json_file:
                json.dump(data, json_file)
            self.logger.get_log().info("[{}] log file save success!".format(case_name))
        except Exception as e:
            print(e)

    # def _log_load(self):
    #     """
    #     保存数据到磁盘
    #     :return:
    #     """
    #     all_case = {}
    #     data = dict()
    #     for i in os.listdir("./log/"):
    #         with open("./log/" + i) as case:
    #             res = case.readline()
    #             api = i.split(".")[0]
    #             data[api] = res
    #     for k, v in data.items():
    #         all_case[k] = {}
    #         # all_case[k]["jid"] = latest_id
    #         all_case[k]["case_name"] = k
    #         all_case[k]["api"] = json.loads(v).get("api")
    #         all_case[k]["result"] = v
    #     return all_case

    def _db_save(self, db, latest_id, log="log"):
        """
        数据库交互
        """
        # db = DB(storage=self.storage)
        latest_case = {}
        data = dict()
        for i in os.listdir("./{}/".format(log)):
            with open("./{}/".format(log) + i) as case:
                res = case.readline()
                api = i.split(".")[0]
                data[api] = res
        for k, v in data.items():
            latest_case["jid"] = latest_id
            latest_case["case_name"] = k
            latest_case["api"] = json.loads(v).get("api")
            latest_case["result"] = v
            db.insert_case(jid=latest_id, data_dict=latest_case, create_time=self.now_time)
