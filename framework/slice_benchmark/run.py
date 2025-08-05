#!/bin/env python3
# -*- coding: utf-8 -*-
# @author Zeref996
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
slice 测试启动
"""

import os
import json
import shutil
from datetime import datetime
import yaml
from time_count_engine import SliceBenchMark
from db.mysql_helper import SliceBenchmarkDB
from db.snapshot import Snapshot
from slicebm_utils.threshold import perf_compare


class SliceTestRun(object):
    """
    slice 测试启动
    """

    def __init__(self):

        self.wheel_link = os.environ.get("SLICE_TEST_WHL", None)
        self.py_version = os.environ.get("SLICE_TEST_PY", None)
        self.framework = os.environ.get("SLICE_BENCHMARK_FRAMEWORKS", "paddle")
        # self.db_config = "apibm_config.yml"
        self.db_config = "/paddle/baidu/paddle/PTSTools/Uploader/apibm_config.yml"
        self.bm = SliceBenchMark()

        if self.framework == "paddle":
            import paddle

            self.cuda = paddle.version.cuda()
            self.commit = paddle.version.commit
        elif self.framework == "torch":
            import torch

            self.cuda = torch.version.cuda
            self.commit = torch.version.git_version
        else:
            raise ValueError(f"not supported framework: {self.framework}")

        snapshot = Snapshot()
        self.md5 = snapshot.get_md5_id()

    def timestamp(self):
        """
        时间戳控制
        """
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def slice_perf(self):
        """
        执行测试用例, 返回测试结果dict
        """
        res_dict = self.bm.perf()
        return res_dict

    def run_test_and_insert_data(self, comment, base):
        """
        插入基线数据
        """
        with open(self.db_config, encoding="utf-8") as f:
            db_config = yaml.load(f, Loader=yaml.FullLoader)
        db = SliceBenchmarkDB(**db_config["Config"]["slice_benchmark"]["MYSQL"])
        data = {
            "comment": comment,
            "env_info": json.dumps({"cuda": self.cuda, "python": self.py_version, "wheel_link": self.wheel_link}),
            "framework": self.framework,
            "status": "running",
            "commit": self.commit,
            "md5_id": self.md5,
            "base": base,
            "create_time": self.timestamp(),
            "update_time": self.timestamp(),
        }
        task_id = db.insert(table="slice_job", data=data)

        res_dict = self.slice_perf()

        fail_cases_list = []
        test_complete_time = self.timestamp()
        for case_name, perf_value in res_dict.items():
            if isinstance(perf_value, float):
                case_table = "slice_case"
            else:
                fail_cases_list.append(case_name)
                case_table = "slice_failcase"

            data = {
                "jid": task_id,
                "case_name": case_name,
                "result": perf_value,
                "create_time": test_complete_time,
            }
            db.insert(table=case_table, data=data)

        update_data = {
            "status": "done",
            "update_time": test_complete_time,
        }
        db.update(table="slice_job", data=update_data, condition=f"id = {task_id}")
        return res_dict, fail_cases_list, task_id

    def get_baseline(self, framework="paddle"):
        """
        获取基线数据
        """
        with open(self.db_config, encoding="utf-8") as f:
            db_config = yaml.load(f, Loader=yaml.FullLoader)
        db = SliceBenchmarkDB(**db_config["Config"]["slice_benchmark"]["MYSQL"])
        baseline_id = db.select_by_condition(
            table="slice_job",
            condition=f"comment = 'slice基线任务' and framework = '{framework}' and status = 'done' and md5_id = '{self.md5}' and base = 1",
        )[-1]["id"]

        baseline_res_dict = db.select_by_condition(table="slice_case", condition=f"jid = {baseline_id}")
        baseline_res_dict = dict([(i["case_name"], i["result"]) for i in baseline_res_dict])
        return baseline_res_dict

    def insert_baseline(self):
        """
        插入基线数据
        """
        self.run_test_and_insert_data(comment="slice基线任务", base=1)

    def ci_test(self):
        """
        ci 测试
        """
        latest_res_dict, fail_cases_list, task_id = self.run_test_and_insert_data(comment="slice测试CI任务", base=0)
        baseline_res_dict = self.get_baseline(framework="paddle")
        print("开始使用本次CI测试结果, 与paddle基线进行性能对比 =============================>")
        perf_compare_res_dict, fail_perf_dict = self.res_dict_compare(baseline_res_dict, latest_res_dict)
        print("已完成paddle基线性能对比 =============================>")

        # 打印torch性能对比信息
        try:
            print("开始使用本次CI测试结果, 与torch基线进行性能对比 =============================>")
            torch_res_dict = self.get_baseline(framework="torch")
            self.torch_res_dict_compare(torch_res_dict, latest_res_dict)
            print("已完成torch基线性能对比 =============================>")
        except Exception as e:
            print(e)
            print("未能完成torch基线性能对比")

        if len(fail_cases_list) > 0:
            print(f"slice测试失败, 存在功能失败case, 失败case有: {fail_cases_list}")
        if len(fail_perf_dict) > 0:
            print(f"slice测试失败, 存在性能下降case, 失败case性能变化: {fail_perf_dict}")

        with open(self.db_config, encoding="utf-8") as f:
            db_config = yaml.load(f, Loader=yaml.FullLoader)
        db = SliceBenchmarkDB(**db_config["Config"]["slice_benchmark"]["MYSQL"])
        if len(fail_cases_list) + len(fail_perf_dict) > 0:
            update_data = {
                "result": "失败",
                "update_time": self.timestamp(),
            }
            db.update(table="slice_job", data=update_data, condition=f"id = {task_id}")
            raise Exception("slice测试失败")
        else:
            update_data = {
                "result": "成功",
                "update_time": self.timestamp(),
            }
            db.update(table="slice_job", data=update_data, condition=f"id = {task_id}")

    def res_dict_compare(self, baseline_res_dict, latest_res_dict):
        """
        性能字典数据对比
        """
        fail_perf_dict = {}
        perf_compare_res_dict = {}
        for case_name, perf_value in latest_res_dict.items():
            if case_name in baseline_res_dict:
                perf_compare_res, grade = perf_compare(baseline_res_dict[case_name], perf_value, case_name)
                if grade == "worse" or grade == "doubt":
                    fail_perf_dict[case_name] = perf_compare_res
                perf_compare_res_dict[case_name] = perf_compare_res
                print(
                    f"{case_name}: 基线数据{baseline_res_dict[case_name]}, 本次测试数据{perf_value}, 相对性能提升{perf_compare_res}, 评分级别{grade}"
                )
            else:
                perf_compare_res_dict[case_name] = "基线数据不存在"
                print(f"{case_name}: 基线数据不存在, 本次测试数据{perf_value}, 无对比值")

        return perf_compare_res_dict, fail_perf_dict

    def torch_res_dict_compare(self, torch_res_dict, latest_res_dict):
        """
        性能字典数据对比
        """
        fail_perf_dict = {}
        perf_compare_res_dict = {}
        for case_name_origin, perf_value in latest_res_dict.items():
            case_name = case_name_origin.replace("paddle", "torch")
            if case_name in torch_res_dict:
                perf_compare_res, grade = perf_compare(torch_res_dict[case_name], perf_value, case_name)
                if grade == "worse" or grade == "doubt":
                    fail_perf_dict[case_name] = perf_compare_res
                perf_compare_res_dict[case_name] = perf_compare_res
                print(
                    f"{case_name}: 基线数据{torch_res_dict[case_name]}, 本次测试数据{perf_value}, 相对性能提升{perf_compare_res}, 评分级别{grade}"
                )
            else:
                perf_compare_res_dict[case_name] = "基线数据不存在"
                print(f"{case_name}: 基线数据不存在, 本次测试数据{perf_value}, 无对比值")

        return perf_compare_res_dict, fail_perf_dict


if __name__ == "__main__":
    test = SliceTestRun()
    # baseline_res_dict = test.get_baseline(framework="paddle")
    # print(baseline_res_dict)

    if os.environ["SLICE_TEST_MODE"] == "insert_baseline":
        test.insert_baseline()
    else:
        test.ci_test()
