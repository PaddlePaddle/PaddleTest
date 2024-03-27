#!/bin/env python3
# -*- coding: utf-8 -*-
# @author Zeref996
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
注意!!!!!!!
修复 self.core_index、self.comment 和 self.wheel_link
"""

import os
import json
import socket
import platform
from datetime import datetime
from db.db import DB
from db.snapshot import Snapshot
from strategy.compare import perf_compare
from tools.logger import Logger
from tools.res_save import xlsx_save


class LayerBenchmarkDB(object):
    """
    layer benchmark 交互模块
    """

    def __init__(self, storage="apibm_config.yaml"):
        """
        :param storage: 信息配置文件
        """
        self.storage = storage
        self.now_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # md5唯一标识码
        self.md5_id = Snapshot().get_md5_id()

        # 效率云环境变量
        self.AGILE_PIPELINE_BUILD_ID = os.environ.get("AGILE_PIPELINE_BUILD_ID", 0)

        # 例行标识
        self.baseline_comment = "baseline_CE_layer_benchmark"
        self.comment = "layer_benchmark_xly_{}".format(self.AGILE_PIPELINE_BUILD_ID)
        self.ci = 0

        # 框架环境信息
        self.framework = os.environ.get("FRAMEWORK")
        self.hardware = os.environ.get("PLT_SET_DEVICE")
        self._frame_info()

        # 机器系统信息
        self.hostname = socket.gethostname()
        self.system = platform.system()

        # 初始化日志
        self.logger = Logger("LayerBenchmarkCE")

    def _frame_info(self):
        """"""
        if os.environ.get("FRAMEWORK") == "PaddlePaddle":
            import paddle

            self.commit = paddle.__git_commit__
            self.version = paddle.__version__
            if os.environ.get("PLT_SET_DEVICE") == "gpu":
                self.env_info = {
                    "cuda": paddle.version.cuda(),
                    "cudnn": paddle.version.cudnn(),
                    "python": os.environ.get("python_ver"),
                    "wheel_link": os.environ.get("wheel_url"),
                    "docker_type": os.environ.get("docker_type"),
                }
            elif os.environ.get("PLT_SET_DEVICE") == "cpu":
                self.env_info = {
                    "python": os.environ.get("python_ver"),
                    "wheel_link": os.environ.get("wheel_url"),
                    "docker_type": os.environ.get("docker_type"),
                }
            else:
                raise Exception("unknown hardware, PaddleLayerTest only support test cpu or gpu")
        elif os.environ.get("FRAMEWORK") == "PyTorch":
            import torch

            self.commit = torch.__git_commit__
            self.version = torch.__version__

            self.env_info = {}
        else:
            raise Exception("unknown framework, PaddleLayerTest only support test PaddlePaddle or Pytorch")

    def latest_insert(self, data_dict, error_list):
        """
        插入最新数据
        """
        db = DB(storage=self.storage)

        # 插入layer_job
        latest_id = db.insert_job(
            comment=self.comment,
            status="running",
            env_info=json.dumps(self.env_info),
            framework=self.framework,
            commit=self.commit,
            version=self.version,
            hostname=self.hostname,
            hardware=self.hardware,
            system=self.system,
            md5_id=self.md5_id,
            base=0,  # 非基线任务
            ci=self.ci,
            create_time=self.now_time,
            update_time=self.now_time,
        )

        # 插入layer_case
        for title, perf_dict in data_dict.items():
            db.insert_case(jid=latest_id, case_name=title, result=json.dumps(perf_dict), create_time=self.now_time)

        if bool(error_list):
            db.update_job(id=latest_id, status="error", update_time=self.now_time)
            print("error cases: {}".format(error_list))
            raise Exception("something wrong with layer benchmark job id: {} !!".format(latest_id))
        else:
            db.update_job(id=latest_id, status="done", update_time=self.now_time)

        # 获取baseline用于对比
        baseline_id = db.select_baseline_job(comment=self.baseline_comment, base=1, ci=self.ci, md5_id=self.md5_id)
        baseline_list = db.select(table="layer_case", condition_list=["jid = {}".format(baseline_id)])

        baseline_dict = {}
        for i in baseline_list:
            baseline_dict[i["case_name"]] = i

        # 性能对比
        compare_dict = {}
        for title, perf_dict in data_dict.items():
            compare_dict[title] = {}
            for perf_engine, t in perf_dict.items():
                compare_dict[title][perf_engine + "_latest"] = t
                compare_dict[title][perf_engine + "_baseline"] = baseline_dict[title][perf_engine]
                compare_dict[title][perf_engine + "_compare"] = perf_compare(
                    baseline=baseline_dict[title][perf_engine], latest=t
                )

        xlsx_save(compare_dict)

    def baseline_insert(self, data_dict, error_list):
        """
        插入最新数据
        """
        db = DB(storage=self.storage)

        # 插入layer_job
        basleine_id = db.insert_job(
            comment=self.baseline_comment,
            status="running",
            env_info=json.dumps(self.env_info),
            framework=self.framework,
            commit=self.commit,
            version=self.version,
            hostname=self.hostname,
            hardware=self.hardware,
            system=self.system,
            md5_id=self.md5_id,
            base=1,  # 基线任务
            ci=self.ci,
            create_time=self.now_time,
            update_time=self.now_time,
        )

        # 插入layer_case
        for title, perf_dict in data_dict.items():
            db.insert_case(jid=basleine_id, case_name=title, result=json.dumps(perf_dict), create_time=self.now_time)

        if bool(error_list):
            db.update_job(id=basleine_id, status="error", update_time=self.now_time)
            print("error cases: {}".format(error_list))
            raise Exception("something wrong with layer benchmark job id: {} !!".format(basleine_id))
        else:
            db.update_job(id=basleine_id, status="done", update_time=self.now_time)
