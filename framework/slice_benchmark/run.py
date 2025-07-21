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
import yaml
from datetime import datetime
from time_count_engine import SliceBenchMark
from db.mysql_helper import SliceBenchmarkDB
from db.snapshot import Snapshot


class SliceTestRun(object):
    """
    slice 测试启动
    """

    def __init__(self):

        self.wheel_link = os.environ.get("SLICE_TEST_WHL", None)
        self.py_version = os.environ.get("SLICE_TEST_PY", None)
        self.framework = os.environ.get("SLICE_TEST_FRAMEWORK", "paddle")
        self.db_config = "apibm_config.yml"
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
        return res_dict, fail_cases_list

    def get_baseline(self):
        """
        获取基线数据
        """
        with open(self.db_config, encoding="utf-8") as f:
            db_config = yaml.load(f, Loader=yaml.FullLoader)
        db = SliceBenchmarkDB(**db_config["Config"]["slice_benchmark"]["MYSQL"])
        baseline_id = db.select_by_condition(
            table="slice_job",
            condition=f"comment = 'slice基线任务' and status = 'done' and md5_id = '{self.md5}' and base = 1",
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
        latest_res_dict, fail_cases_list = self.run_test_and_insert_data(comment="slice测试CI任务", base=0)
        baseline_res_dict = self.get_baseline()
        perf_compare_res_dict, fail_perf_dict = self.res_dict_compare(baseline_res_dict, latest_res_dict)

        if len(fail_cases_list) > 0:
            print(f"slice测试失败, 存在功能失败case, 失败case有: {fail_cases_list}")
        if len(fail_perf_dict) > 0:
            print(f"slice测试失败, 存在性能下降case, 失败case性能变化: {fail_perf_dict}")
        if len(fail_cases_list) + len(fail_perf_dict) > 0:
            raise Exception("slice测试失败")

    def perf_grade(self, res):
        """
        评分标准
        :param res: 性能对比结果
        :return:
        """
        grade = ""
        if isinstance(res, str):
            grade = res
        else:
            if res <= -0.2:
                grade = "worse"
            elif -0.2 < res <= -0.1:
                grade = "doubt"
            elif -0.1 < res <= 0.1:
                grade = "equal"
            elif res > 0.1:
                grade = "better"
        return grade

    def perf_compare(self, baseline, latest):
        """
        比较函数
        :param latest: 待测值
        :param baseline: 基线值
        :return: 比例值
        """
        if isinstance(baseline, str) or isinstance(baseline, str):
            res = "error"
            return res
        else:
            if baseline == 0 or latest == 0:
                res = 0
            else:
                if latest > baseline:
                    res = (latest - baseline) / baseline * -1
                else:
                    res = (baseline - latest) / latest
        grade = self.perf_grade(res)
        return res, grade
        # return "{:.2f}%".format(res * 100)

    def res_dict_compare(self, baseline_res_dict, latest_res_dict):
        """
        性能字典数据对比
        """
        fail_perf_dict = {}
        perf_compare_res_dict = {}
        for case_name, perf_value in latest_res_dict.items():
            if case_name in baseline_res_dict:
                perf_compare_res, grade = self.perf_compare(baseline_res_dict[case_name], perf_value)
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


if __name__ == "__main__":
    test = SliceTestRun()
    # baseline_res_dict = test.get_baseline()
    # print(baseline_res_dict)

    # test.insert_baseline()

    test.ci_test()
