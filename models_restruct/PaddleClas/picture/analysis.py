# encoding: utf-8
"""
分析对比绘图
"""
import os
import re
from email.mime import base
from collections import defaultdict
import argparse
import logging
import traceback
import time
import tarfile
import shutil
import yaml
import wget
import numpy as np
import matplotlib.pyplot as plt

logging.getLogger().setLevel(logging.INFO)

LOG_FILE_SUFFIX = ".log"
BASE_MODE = "train_dy2st"


def parse_args():
    """
    传入参数
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, help="The path of config file.", default="resnet_prim_cinn.yaml")
    parser.add_argument("--log_dir", type=str, help="The dir of log files.", default="./")
    args = parser.parse_args()
    return args


class LogInfo:
    """
    LogInfo
    """

    def __init__(self, mode, epoch):
        """
        __init__
        """
        self.mode = mode
        self.epoch = epoch
        self.targes = defaultdict(list)

    def keys(self):
        """
        keys
        """
        return self.targes.keys()

    def values(self):
        """
        values
        """
        return self.targes.values()

    def get(self, name):
        """
        get
        """
        return self.targes[name]

    # get end step for each epoch
    def target_from_key_steps(self, target):
        """
        target_from_key_steps
        """
        total_step = len(self.get(target))
        step_gap = int(total_step / self.epoch)
        return self.get(target)[step_gap - 1 : total_step : step_gap]

    def __str__(self) -> str:
        """
        __str__
        """
        res = ""
        for k, v in self.targes.items():
            res += "{}: {}\n".format(k, v)
        return res


def calc_percent(actual, desired):
    """
    calc_percent
    """
    res = []
    for a, d in zip(actual, desired):
        # if a is zero
        if not a:
            a = a + 1e-8
            d = d + 1e-8
        res.append((d - a) / a)
    return res


def analysis(log_dir, config, train_or_eval):
    """
    analysis
    """
    if train_or_eval not in config.keys():
        return None

    modes = config["modes"].split(";")
    epoch = config["epoch"]
    targets = config[train_or_eval]["targets"].split(";")
    keywords = config[train_or_eval]["keywords"].split(";")

    # construc pattern accoring to targets
    # pattern = '.*top1: (?P<top1>\d+(\.\d+)?).*, .*top5: (?P<top5>\d+(\.\d+)?).*, .*loss: (?P<loss>\d+(\.\d+)?)'
    pattern = ""
    for t in targets:
        pattern = r".*{}: (?P<{}>\d+(\.\d+)?).*".format(t, t)
    pattern = pattern.rstrip(".*, ")

    log_info_map = {}
    for mode in modes:
        log_info_map[mode] = LogInfo(mode, epoch)

    for mode, log_info in log_info_map.items():
        log_path = os.path.join(log_dir, log_info.mode) + LOG_FILE_SUFFIX
        with open(log_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                if all([word in line for word in keywords]):
                    res = re.search(pattern, line)
                    for t in targets:
                        log_info.get(t).append(float(res.group(t)))

    return log_info_map


def draw_line(actual_y, desired_y, actual_mode, desired_mode, target, train_or_eval):
    """
    draw_line
    """
    plt.figure()
    plt.plot([i for i in range(len(desired_y))], desired_y, color="g", label=desired_mode)
    plt.plot([i for i in range(len(actual_y))], actual_y, color="r", label=actual_mode)
    plt.legend()
    if train_or_eval == "train":
        plt.xlabel("iter")
    else:
        plt.xlabel("epoch")
    plt.title(target)
    plt.savefig("[{}][{}]--[{}]vs[{}]".format(train_or_eval, target, desired_mode, actual_mode))
    plt.close()


def draw(log_info_map, config, train_or_eval):
    """
    draw
    """
    BASE_MODE = config["base_mode"]
    if train_or_eval not in config.keys():
        return None
    base_log_info = log_info_map[BASE_MODE]
    for mode, log_info in log_info_map.items():
        if mode == BASE_MODE:
            continue
        for target in config[train_or_eval]["targets"].split(";"):
            draw_line(base_log_info.get(target), log_info.get(target), BASE_MODE, mode, target, train_or_eval)
    return time.strftime("_%Y_%m_%d", time.gmtime(time.time()))


def show_step_data(log_info_map, config, train_or_eval):
    """
    show_step_data
    """
    BASE_MODE = config["base_mode"]
    if train_or_eval not in config.keys():
        return None
    base_log_info = log_info_map[BASE_MODE]
    target_result = {}
    for target in config[train_or_eval]["targets"].split(";"):
        # print('###target', target)
        table = {}
        if train_or_eval == "train":
            actual_step_target = base_log_info.target_from_key_steps(target)
        elif train_or_eval == "eval":
            actual_step_target = base_log_info.get(target)

        for mode, log_info in log_info_map.items():
            if train_or_eval == "train":
                desired_step_target = log_info.target_from_key_steps(target)
            elif train_or_eval == "eval":
                desired_step_target = log_info.get(target)

            # 表格中最多只显示20条数据
            table["[{}][{}]".format(target, mode)] = desired_step_target[-20:]
            if mode == BASE_MODE:
                continue

            if target in ["top1", "top5"]:
                diff = calc_percent(actual_step_target, desired_step_target)
                diff_all = []
                for num in diff:
                    diff_all.append("%.3f%%" % (num * 100))
            else:
                diff = list(map(lambda a, d: d - a, actual_step_target, desired_step_target))
                diff_all = []
                for num in diff:
                    diff_all.append("%.5f" % (num))
            # 表格中最多只显示20条数据
            table["[{}] vs [{}]".format(mode, BASE_MODE)] = diff_all[-20:]

        # for k, v in table.items():
        #     print(k)
        #     print(v)
        # print(train_or_eval + '-'*200)
        target_result[target] = table
    return target_result


if __name__ == "__main__":
    args = parse_args()
    f = open(args.config, "r")
    config = yaml.load(f, Loader=yaml.FullLoader)
    f.close()

    train_or_eval = "train"
    train_log_info_map = analysis(args.log_dir, config, train_or_eval)
    draw(train_log_info_map, config, train_or_eval)
    train_target_result = show_step_data(train_log_info_map, config, train_or_eval)

    train_or_eval = "eval"
    if train_or_eval in config.keys():
        eval_log_info_map = analysis(args.log_dir, config, train_or_eval)
        draw(eval_log_info_map, config, train_or_eval)
        eval_target_result = show_step_data(eval_log_info_map, config, train_or_eval)
