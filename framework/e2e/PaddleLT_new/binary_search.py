#!/bin/env python3
# -*- coding: utf-8 -*-
# @author Zeref996
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
二分工具
"""
import os
import ast
import sys
import argparse
import subprocess
import numpy as np
from layertest import LayerTest
from strategy.compare import perf_compare
from tools.logger import Logger


def get_commits(start, end):
    """
    get all the commits in search interval
    """
    print("start:{}".format(start))
    print("end:{}".format(end))
    cmd = "git log {}..{} --pretty=oneline".format(start, end)
    log = subprocess.getstatusoutput(cmd)
    print(log[1])
    commit_list = []
    candidate_commit = log[1].split("\n")
    print(candidate_commit)
    for commit in candidate_commit:
        commit = commit.split(" ")[0]
        print("commit:{}".format(commit))
        commit_list.append(commit)
    return commit_list


class BinarySearch(object):
    """
    性能/精度通用二分定位工具
    """

    def __init__(self, commit_list, title, layerfile, testing, perf_decay=None, test_obj=LayerTest):
        """
        初始化
        commit_list: 二分定位commit的范围list[commit]
        title: 日志标题, 随便取
        layerfile: 子图路径, 例如./layercase/sublayer1000/Det_cases/ppyolo_ppyolov2_r50vd_dcn_365e_coco/SIR_76.py
        testing: 测试yaml路径, 例如 yaml/dy^dy2stcinn_eval_benchmark.yml
        perf_decay: 仅用于性能, 某个engine名称+预期耗时+性能下降比例, 组成的list, 例如["dy2st_eval_cinn_perf", 0.0635672, -0.3]

        """
        self.logger = Logger("PLT二分定位")

        self.commit_list = commit_list

        self.title = title
        self.layerfile = layerfile
        self.testing = testing
        self.perf_decay = perf_decay
        self.test_obj = test_obj
        self.py_cmd = os.environ.get("python_ver")
        self.testing_mode = os.environ.get("TESTING_MODE")

    def _status_print(self, exit_code, status_str):
        """
        状态打印
        """
        if exit_code == 0:
            self.logger.get_log().info(f"{status_str} successfully")
        else:
            self.logger.get_log().error(f"{status_str} failed")
            sys.exit(-1)

    def _install_paddle(self, commit_id):
        """
        安装 paddle
        """
        exit_code = os.system(f"{self.py_cmd} -m pip uninstall paddlepaddle-gpu -y")
        self._status_print(exit_code=exit_code, status_str="uninstall paddlepaddle-gpu")

        whl_link = (
            "https://paddle-qa.bj.bcebos.com/paddle-pipe"
            "line/Develop-GpuAll-LinuxCentos-Gcc82-Cuda112-Trtoff-Py38-Compile/{}/paddle"
            "paddle_gpu-0.0.0-cp38-cp38-linux_x86_64.whl".format(commit_id)
        )
        exit_code = os.system(f"{self.py_cmd} -m pip install {whl_link}")
        self._status_print(exit_code=exit_code, status_str="install paddlepaddle-gpu")
        self.logger.get_log().info("commit {} install done".format(commit_id))
        return 0

    def _precision_debug(self, commit_id):
        """
        精度debug
        """
        exc = 0
        try:
            self.test_obj(title=self.title, layerfile=self.layerfile, testing=self.testing)._case_run()
        except Exception:
            exc += 1

        if exc > 0:
            self.logger.get_log().info(f"执行失败commit: {commit_id}")
            return False
        else:
            self.logger.get_log().info(f"执行成功commit: {commit_id}")
            return True

    def _performance_debug(self, commit_id):
        """
        精度debug
        """
        res_dict, exc = self.test_obj(title=self.title, layerfile=self.layerfile, testing=self.testing)._perf_case_run()
        latest = res_dict[self.perf_decay[0]]
        baseline = self.perf_decay[1]
        decay_rate = self.perf_decay[2]

        compare_res = perf_compare(baseline, latest)
        fluctuate_rate = 0.15
        if exc > 0 or compare_res < decay_rate - fluctuate_rate:
            self.logger.get_log().info(f"执行失败commit: {commit_id}")
            return False
        else:
            self.logger.get_log().info(f"执行成功commit: {commit_id}")
            return True

    def _commit_locate(self, commits):
        """
        commit定位
        """
        self.logger.get_log().info("测试case名称: {}".format(self.title))

        if len(commits) == 2:
            self.logger.get_log().info(
                "only two candidate commits left in binary_search, the final commit is {}".format(commits[0])
            )
            return commits[0]
        left, right = 0, len(commits) - 1
        if left <= right:
            mid = left + (right - left) // 2
            commit = commits[mid]

            self._install_paddle(commit)

            if eval(f"self._{self.testing_mode}_debug")(commit):
                self.logger.get_log().info("the commit {} is success".format(commit))
                self.logger.get_log().info("mid value:{}".format(mid))
                selected_commits = commits[: mid + 1]
                res = self._commit_locate(selected_commits)
            else:
                self.logger.get_log().info("the commit {} is failed".format(commit))
                selected_commits = commits[mid:]
                res = self._commit_locate(selected_commits)
        return res


if __name__ == "__main__":
    cur_path = os.getcwd()
    os.system("rm -rf paddle && git clone -b develop http://github.com/paddlepaddle/paddle.git")
    os.chdir(os.path.join(cur_path, "paddle"))
    start_commit = "9d5f31687cce16a976256723a70df4550085d685"  # 成功commit
    end_commit = "7139309b30f65c8bb8fb0e427b194c265e955c87"  # 失败commit
    commits = get_commits(start=start_commit, end=end_commit)
    print("the candidate commits is {}".format(commits))
    os.chdir(cur_path)

    final_commit = BinarySearch(
        commit_list=commits,
        title="PrecisionBS",
        layerfile="./layercase/perf_monitor/manual_subgraph/layernorm_fp32_shape_1_13_4096.py",
        testing="yaml/dy^dy2stcinn_eval_benchmark.yml",
        perf_decay=None,  # ["dy2st_eval_cinn_perf", 0.042814, -0.3]
        test_obj=LayerTest,
    )._commit_locate(commits)
    print("the pr with problem is {}".format(final_commit))
    f = open("final_commit.txt", "w")
    f.writelines("the final commit is:{}".format(final_commit))
    f.close()
