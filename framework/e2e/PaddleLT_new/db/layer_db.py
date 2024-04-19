#!/bin/env python3
# -*- coding: utf-8 -*-
# @author Zeref996
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
子图db交互模块
"""

import os
import json
import socket
import platform
from datetime import datetime
from db.db import DB
from db.snapshot import Snapshot

# from strategy.compare import perf_compare
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
        self.AGILE_PIPELINE_BUILD_ID = int(os.environ.get("AGILE_PIPELINE_BUILD_ID", 0))

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

        # 子图种类信息
        self.layer_type = os.environ.get("CASE_TYPE")

        # 初始化日志
        self.logger = Logger("LayerBenchmarkDB")

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
            agile_pipeline_build_id=self.AGILE_PIPELINE_BUILD_ID,
            layer_type=self.layer_type,
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
        # 保存job id到txt
        with open("job_id.txt", "w") as file:
            file.write(str(latest_id))
        self.logger.get_log().info("性能测试job_id: {}".format(latest_id))

        # 插入layer_case
        for title, perf_dict in data_dict.items():
            db.insert_case(jid=latest_id, case_name=title, result=json.dumps(perf_dict), create_time=self.now_time)

        if bool(error_list):
            db.update_job(id=latest_id, status="error", update_time=self.now_time)
            self.logger.get_log().warn("error cases: {}".format(error_list))
            # print("error cases: {}".format(error_list))
        else:
            db.update_job(id=latest_id, status="done", update_time=self.now_time)

        # self.compare_with_baseline(data_dict=data_dict, error_list=error_list)
        # baseline_dict = self.get_baseline_dict()

        # compare_dict = perf_compare_dict(baseline_dict=baseline_dict, data_dict=data_dict, error_list=error_list)

        if bool(error_list):
            raise Exception("something wrong with layer benchmark job id: {} !!".format(latest_id))

        # return data_dict

    def get_baseline_dict(self):
        """
        获取baseline dict
        """
        # 获取baseline用于对比
        db = DB(storage=self.storage)
        baseline_job = db.select_baseline_job(comment=self.baseline_comment, base=1, ci=self.ci, md5_id=self.md5_id)
        baseline_id = baseline_job["id"]
        baseline_layer_type = baseline_job["layer_type"]
        baseline_list = db.select(table="layer_case", condition_list=["jid = {}".format(baseline_id)])
        baseline_dict = {}
        for i in baseline_list:
            baseline_dict[i["case_name"]] = i
        return baseline_dict, baseline_layer_type

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
            agile_pipeline_build_id=self.AGILE_PIPELINE_BUILD_ID,
            layer_type=self.layer_type,
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
        # 保存job id到txt
        with open("job_id.txt", "w") as file:
            file.write(str(basleine_id))
        self.logger.get_log().info("性能测试job_id: {}".format(basleine_id))

        # 插入layer_case
        for title, perf_dict in data_dict.items():
            db.insert_case(jid=basleine_id, case_name=title, result=json.dumps(perf_dict), create_time=self.now_time)

        if bool(error_list):
            db.update_job(id=basleine_id, status="error", update_time=self.now_time)
            self.logger.get_log().warn("error cases: {}".format(error_list))
            # print("error cases: {}".format(error_list))
            raise Exception("something wrong with layer benchmark job id: {} !!".format(basleine_id))
        else:
            db.update_job(id=basleine_id, status="done", update_time=self.now_time)

    # def compare_with_baseline(self, data_dict, error_list):
    #     """
    #     与基线对比
    #     """
    #     db = DB(storage=self.storage)
    #     # 获取baseline用于对比
    #     baseline_id = db.select_baseline_job(comment=self.baseline_comment, base=1, ci=self.ci, md5_id=self.md5_id)
    #     baseline_list = db.select(table="layer_case", condition_list=["jid = {}".format(baseline_id)])

    #     baseline_dict = {}
    #     for i in baseline_list:
    #         baseline_dict[i["case_name"]] = i

    #     # 性能对比
    #     # compare_dict = {}
    #     # for title, perf_dict in data_dict.items():
    #     #     if title not in error_list:
    #     #         compare_dict[title] = {}
    #     #         for perf_engine, t in perf_dict.items():
    #     #             compare_dict[title][perf_engine + "_latest"] = t
    #     #             compare_dict[title][perf_engine + "_baseline"] = json.loads(baseline_dict[title]["result"])[
    #     #                 perf_engine
    #     #             ]
    #     #             compare_dict[title][perf_engine + "_compare"] = perf_compare(
    #     #                 baseline=json.loads(baseline_dict[title]["result"])[perf_engine], latest=t
    #     #             )
    #     compare_dict = {}
    #     for title, perf_dict in data_dict.items():
    #         if title not in error_list:
    #             compare_dict[title] = {}
    #             # path = "layercase/sublayer/aspdiaopi"
    #             # new_content = "newdir"
    #             # # 使用 split() 方法将路径分割成两部分，并取第一个部分
    #             head, tail = title.split('^', 1)
    #             # 使用 join() 方法将新的内容和原路径的第二部分拼接起来
    #             baseline_title = '^'.join(["layercase", tail])
    #             for perf_engine, t in perf_dict.items():
    #                 compare_dict[title][perf_engine + "_latest"] = t
    #                 compare_dict[title][perf_engine + "_baseline"] =
    #                   json.loads(baseline_dict[baseline_title]["result"])[
    #                     perf_engine
    #                 ]
    #                 compare_dict[title][perf_engine + "_compare"] = perf_compare(
    #                     baseline=json.loads(baseline_dict[baseline_title]["result"])[perf_engine], latest=t
    #                 )

    #     # 生成表格
    #     xlsx_save(
    #         sublayer_dict=compare_dict,
    #         excel_file=os.environ.get("TESTING").replace("yaml/", "").replace(".yml", "") + ".xlsx",
    #     )
