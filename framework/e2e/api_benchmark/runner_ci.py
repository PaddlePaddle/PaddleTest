#!/bin/env python
# -*- coding: utf-8 -*-
# @author DDDivano
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
runner
"""

import argparse
import platform
import sys

import paddle

sys.path.append("..")
from utils.yaml_loader import YamlLoader
from utils.logger import logger
from benchtrans import BenchTrans
from jelly_v2 import Jelly_v2
from tools import delete


SKIP_DICT = {"Windows": ["fft"], "Darwin": ["fft"], "Linux": []}
INDEX_DICT = {}
SPECIAL = False  # speacial for emergency


class ApiBenchmarkCI(object):
    """
    api benchmark 调度CI, 监控cpu+前向, 支持多个机器baseline
    """

    def __init__(self, baseline, yaml_path, place="cpu", enable_backward=0):
        """
        :param baseline: 性能baseline键值对, key为case名, value为性能float
        """
        self.baseline_dict = baseline
        self.yaml_path = yaml_path
        self.place = place
        self.enable_backward = enable_backward
        self.forward_time_dict = {}

    def run_cases(self):
        """
        例行任务，调试任务二次运行调用。 一定入库，存在数据库操作
        """
        yaml_file = self.yaml_path
        yaml_loader = YamlLoader(yaml_file)

        cases_name = yaml_loader.get_all_case_name()
        for case_name in cases_name:
            # Skip cases
            if case_name in SKIP_DICT[platform.system()]:
                logger.get_log().warning("skip case -->{}<--".format(case_name))
                continue
            if SPECIAL and case_name not in SKIP_DICT[platform.system()]:
                logger.get_log().warning("case is not in index_dict, skipping...-->{}<--".format(case_name))
                continue
            if not case_name.endswith("_0"):
                logger.get_log().warning("skip case -->{}<--".format(case_name))
                continue
            # if yaml_info == "case_0":
            #     if not case_name.endswith("_0"):
            #         logger.get_log().warning("skip case -->{}<--".format(case_name))
            #         continue
            # if yaml_info == "case_1":
            #     if not case_name.endswith("_1"):
            #         logger.get_log().warning("skip case -->{}<--".format(case_name))
            #         continue
            # if yaml_info == "case_2":
            #     if not case_name.endswith("_2"):
            #         logger.get_log().warning("skip case -->{}<--".format(case_name))
            #         continue
            case_info = yaml_loader.get_case_info(case_name)
            try:
                bt = BenchTrans(case_info)
                # if enable_backward == 0:
                #     enable_backward_trigger = False
                # else:
                #     enable_backward_trigger = bt.enable_backward()
                enable_backward_trigger = False

                api = bt.get_paddle_api()
                jelly = Jelly_v2(
                    api=api,
                    framework="paddle",
                    title=case_name,
                    place="cpu",
                    card=None,
                    default_dtype="float32",
                    enable_backward=enable_backward_trigger,
                )
                jelly.set_paddle_param(bt.get_paddle_inputs(), bt.get_paddle_param())
                jelly.set_paddle_method(bt.get_paddle_method())
                forward_time = jelly._return_forward()
                print("forward_time is: ", forward_time)
                print("forward_time type is: ", type(forward_time))
                self.forward_time_dict[case_name] = forward_time
                print("forward_time_dict is: ", self.forward_time_dict)

            except Exception as e:
                # 存储异常
                self.forward_time_dict[case_name] = e
                paddle.enable_static()
                paddle.disable_static()
                logger.get_log().warn(e)
                # print('forward_time_dict type is: ', forward_time_dict)
        return self.forward_time_dict

    def compare(self):
        """

        :return:
        """
        result_dict = {}
        latest_time_dict = self.run_cases()
        baseline_time_dict = self.baseline_dict
        for k, v in latest_time_dict.items():
            baseline_time = baseline_time_dict[k]
            if isinstance(v, float):
                if v > baseline_time:
                    res = (v / baseline_time) * -1
                else:
                    res = baseline_time / v
            else:
                res = v
            result_dict[k] = res
        return result_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    # parser.add_argument("--id", type=int, default=0, help="job id")
    parser.add_argument("--yaml", type=str, help="input the yaml path")
    # parser.add_argument(
    #     "--mode",
    #     type=str,
    #     default="testing",
    #     help="""choose mode: [schedule] for ce or [testing] for native test or [rerun] for double check.
    #           [schedule] and [rerun] will write data into database """,
    # )
    # parser.add_argument("--routine", type=int, default=1, help="if 1, daily routine mission")
    # parser.add_argument("--framework", type=str, default="paddle", help="[paddle] | [torch] | [all]")
    # parser.add_argument("--wheel_link", type=str, default="not_yet", help="paddle wheel link")
    # parser.add_argument("--case", type=str, default="Tanh", help="case name for [testing] and [rerun] mode")
    # parser.add_argument("--place", type=str, default="cpu", help="[cpu] or [gpu]")
    # parser.add_argument("--cuda", type=str, default=None, help="cuda version like v10.2 | v11.2 etc.")
    # parser.add_argument("--cudnn", type=str, default=None, help="cudnn version like v7.6.5 etc.")
    # parser.add_argument("--card", type=str, default=None, help="card number , default is 0")
    # parser.add_argument("--comment", type=str, default=None, help="your comment")
    # parser.add_argument("--yaml_info", type=str, default=None, help="[case_0] or [case_1] or [case_2]")
    # parser.add_argument("--enable_backward", type=int, default=1, help="if 1, enable backward test")
    # parser.add_argument("--python", type=str, default=None, help="python version like python3.7 | python3.8 etc.")
    # parser.add_argument("--storage", type=str, default=None, help="path of storage.yaml")
    args = parser.parse_args()

    # # 验证参数组合正确性
    # if args.place == "gpu" and (args.cuda is None or args.cudnn is None):
    #     logger.get_log().error("GPU情况下必须输入cuda和cudnn版本")
    #     raise AttributeError

    # 判断log文件是否干净
    delete("./log")

    # baseline = {'Tanh_0': 0.00431584, 'conv2d_0': 0.022992369}
    baseline = {"Tanh_0": 0.00331584, "conv2d_0": 0.042992369}
    apibm = ApiBenchmarkCI(baseline, args.yaml)
    result_dict = apibm.compare()
    print("result_dict is :", result_dict)
