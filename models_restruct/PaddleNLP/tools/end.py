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
        ]
        self.reponame = os.environ["reponame"]
        self.qa_yaml_name = os.environ["qa_yaml_name"]
        self.TRAIN_LOG_PATH = os.path.join("logs", self.reponame, self.qa_yaml_name)

    def drow_picture(self, model_name, baseline_info, strategy_info):
        """drowing loss curve"""
        logger.info("strategy_info {}".format(strategy_info))
        num = 1
        for key, value in strategy_info.items():
            plt.subplot(1, len(strategy_info.items()), num)
            plt.plot(
                [i for i in range(len(baseline_info["baseline"]))],
                baseline_info["baseline"],
                color="g",
                label="baseline",
            )
            plt.plot([i for i in range(len(baseline_info["baseline"]))], value, color="r", label=key)

            plt.legend()
            if  num == 1:
                plt.xlabel("step")
                plt.ylabel("loss")
                picture_name = model_name.lstrip('model_zoo^').capitalize()
                plt.title(picture_name)
            num = num + 1
        if not os.path.exists("picture"):
            os.makedirs("picture")
        plt.savefig("./picture/{}.png".format(self.pipeline_name))
        plt.close()

    def get_metrics(self, filename, kpi):
        """
        get metrics:
        loss acc ips ...
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
        strategy_info = {}
        for file in os.listdir(self.TRAIN_LOG_PATH):
            logger.info("check log file is {}".format(file))
            if re.compile("baseline").findall(file):
                baseline_info["baseline"] = self.get_metrics(self.TRAIN_LOG_PATH+'/'+file, "loss")
            else:
                strategy = file.split("train_")[-1].replace(".log", "")
                strategy_info[strategy] = self.get_metrics(self.TRAIN_LOG_PATH+'/'+file, "loss")

        self.drow_picture(self.qa_yaml_name, baseline_info, strategy_info)

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
    model = PaddleNLP_End()
    model.build_end()
    return 0


if __name__ == "__main__":
    run()
