# encoding: utf-8
"""
post preocessing
"""
from copyreg import pickle
import os
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
        self.pipeline_name = os.environ[
            "AGILE_PIPELINE_NAME"
        ]  # PaddleNLP-LinuxConvergence-Cuda112-Python38-GPT-Develop
        self.reponame = os.environ["reponame"]
        self.qa_yaml_name = os.environ["qa_yaml_name"]
        self.case_step = os.environ["case_step"]
        self.case_name = os.environ["case_name"]
        self.TRAIN_LOG_PATH = os.path.join("logs", self.reponame, self.qa_yaml_name, "train_")

    def drow_picture(self, model_name, baseline_info, stategy_info):
        """drowing loss curve"""
        num = 1
        for key, value in stategy_info.items():
            plt.subplot(1, len(stategy_info.items()), num)
            plt.plot(
                [i for i in range(len(baseline_info["baseline"]))],
                baseline_info["baseline"],
                color="g",
                label="baseline",
            )
            plt.plot([i for i in range(len(baseline_info["baseline"]))], value, color="r", label=key)

            plt.legend()
            picture_name = model_name + "_baseline_{}".format(key)
            plt.title(picture_name)
            num = num + 1
        plt.savefig("./picture/{}.png".format(self.pipeline_name))
        plt.close()

    def get_metrics(self, filename, kpi):
        """
        get metrics:
        loss acc ips ...
        """
        kpi_value = -1
        f = open(filename, encoding="utf-8", errors="ignore")
        for line in f.readlines():
            if kpi + ":" in line:
                regexp = r"%s:(\s*\d+(?:\.\d+)?)" % kpi
                r = re.findall(regexp, line)
                # 如果解析不到符合格式到指标，默认值设置为-1
                kpi_value = float(r[0].strip()) if len(r) > 0 else -1
                print(kpi_value)
        f.close()
        return kpi_value

    def analysis_log(self):
        """
        Analysis log & save it to ./picture/

        Return: dict-> log_info
        Examples:
            log_info={
            'model_name': 'bert',
            'baseline': [], # loss
            'cinn': [], # loss
             ...
            }
        """
        baseline_info = {}
        stategy_info = {}
        for file in os.listdir(self.TRAIN_LOG_PATH):
            if re.compile("baseline").findall(file):
                baseline_info["baseline"] = self.get.metrics(file, "loss")
            else:
                strategy = file.split("train_")[-1].replace(".log", "")
                stategy_info[strategy] = self.get.metrics(file, "loss")

        self.drow_picture(self.qa_yaml_name, baseline_info, stategy_info)

    def build_end(self):
        """
        执行准备过程
        """
        # 进入repo中
        logger.info("build remove_data start")
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
    model = PaddleNLP_End()
    model.build_end()
    return 0


if __name__ == "__main__":
    run()
