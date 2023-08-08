# encoding: utf-8
"""
post preocessing
"""
from copyreg import pickle
import os
from platform import system
import pstats
import sys
import json
import glob
import shutil
import argparse
import logging
import re
import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger("ce")


class PaddleNLP_End(object):
    """
    post processing
    """

    def __init__(self):
        """
        init
        """
        # export qa_yaml_name='model_zoo^bert_convergence_dy2st'
        # export reponame='PaddleNLP'
        # export system='linux_convergence'
        self.reponame = os.environ["reponame"]
        self.qa_yaml_name = os.environ["qa_yaml_name"]
        self.TRAIN_LOG_PATH = os.path.join("logs", self.reponame, self.qa_yaml_name)

    def drow_picture(self, model_name, baseline_info, strategy_info, metrics):
        """drowing loss/ips curve"""
        num = 1
        picture_name = (model_name.replace("model_zoo^", "") + "_" + metrics).upper()
        for key, value in strategy_info.items():
            if re.compile(metrics).findall(key):
                plt.subplot(1, len(strategy_info.items()) // 2, num)
                plt.plot(
                    [i for i in range(len(baseline_info["baseline_" + metrics]))],
                    baseline_info["baseline_" + metrics],
                    color="g",
                    label="baseline_" + metrics,
                )
                plt.plot([i for i in range(len(baseline_info["baseline_" + metrics]))], value, color="r", label=key)

                plt.legend()
                if num == 1:
                    plt.xlabel("step")
                    plt.ylabel(metrics)
                    plt.title(picture_name)
                num = num + 1
        if not os.path.exists("picture"):
            os.makedirs("picture")
        plt.savefig("./picture/{}.png".format(picture_name))
        plt.close()

    def get_metrics(self, filename, kpi):
        """
        Get metrics such as: loss acc ips ...
        """
        data_list = []
        f = open(filename, encoding="utf-8", errors="ignore")
        for line in f.readlines():
            if kpi + ":" in line:
                regexp = r"%s:(\s*\d+(?:\.\d+)?)" % kpi
                r = re.findall(regexp, line)
                # 如果解析不到符合格式到指标，默认值设置为-1
                kpi_value = float(r[0].strip()) if len(r) > 0 else -1
                data_list.append(kpi_value)
        f.close()
        return data_list

    def analysis_log(self):
        """
        Analysis log & save it to ./picture/

        Return:
        Examples:
            baseline_info={
            'baseline_loss': [], # loss
            'baseline_ips': [], # ips
             ...
            }
            strategy_info={
            'cinn_loss': [], # loss
            'cinn_ips': [], # ips
             ...
            }

        """
        baseline_info = {}
        strategy_info = {}
        for file in os.listdir(self.TRAIN_LOG_PATH):
            logger.info("check log file is {}".format(file))
            if re.compile("baseline").findall(file):
                baseline_info["baseline_loss"] = self.get_metrics(self.TRAIN_LOG_PATH + "/" + file, "loss")
                baseline_info["baseline_ips"] = self.get_metrics(self.TRAIN_LOG_PATH + "/" + file, "ips")
            elif re.compile("dy2st").findall(file):
                strategy_loss = file.split("train_")[-1].replace(".log", "") + "_loss"
                strategy_ips = file.split("train_")[-1].replace(".log", "") + "_ips"
                strategy_info[strategy_loss] = self.get_metrics(self.TRAIN_LOG_PATH + "/" + file, "loss")
                strategy_info[strategy_ips] = self.get_metrics(self.TRAIN_LOG_PATH + "/" + file, "ips")
            else:
                logger.info("this log file not convergence task ")

        self.drow_picture(self.qa_yaml_name, baseline_info, strategy_info, metrics="loss")
        self.drow_picture(self.qa_yaml_name, baseline_info, strategy_info, metrics="ips")

    def build_end(self):
        """
        执行准备过程
        """
        # 进入repo中
        logger.info("build analysis log start")
        ret = 0
        ret = self.analysis_log()
        if ret:
            logger.info("build analysis log failed")
            return ret
        logger.info("build analysis log end")
        return ret


def run():
    """
    执行入口
    """
    platform = os.environ["system"]
    all = re.compile("All").findall(os.environ["AGILE_PIPELINE_NAME"])
    if platform == "linux_convergence" and not all:
        model = PaddleNLP_End()
        model.build_end()
        return 0
    else:
        return 0


if __name__ == "__main__":
    run()
