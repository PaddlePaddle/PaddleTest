# encoding: utf-8
"""
执行case前：生成yaml，设置特殊参数，改变监控指标
"""
import os
import sys
import json
import shutil
import logging
import tarfile
import argparse
import yaml
import wget
import numpy as np

logger = logging.getLogger("ce")


class PaddleNLP_Case_Start(object):
    """
    自定义环境准备
    """

    def __init__(self):
        """
        初始化变量
        """
        self.reponame = os.environ["reponame"]
        self.mode = os.environ["mode"]
        self.case_step = os.environ["case_step"]
        self.case_name = os.environ["case_name"]
        self.qa_yaml_name = os.environ["qa_yaml_name"]

    def build_prepare(self):
        """
        执行准备过程
        """
        if "convergence" in self.qa_yaml_name:
            logger.info("convergence tag is: {}".format(self.case_name.split("train_")[-1]))

            os.environ["NVIDIA_TF32_OVERRIDE"] = "1"
            os.environ["FLAGS_cudnn_deterministic"] = "1"

            logger.info("export NVIDIA_TF32_OVERRIDE=1")
            logger.info("export FLAGS_cudnn_deterministic=1")

            if self.case_name.split("train_")[-1] == "dy2st_cinn":
                os.environ["FLAGS_use_cinn"] = "1"
                os.environ["FLAGS_deny_cinn_ops"] = "dropout"
            elif self.case_name.split("train_")[-1] == "dy2st_prim":
                os.environ["FLAGS_prim_all"] = "true"
            elif self.case_name.split("train_")[-1] == "dy2st_prim_cinn":
                os.environ["FLAGS_use_cinn"] = "1"
                os.environ["FLAGS_prim_all"] = "true"
                os.environ["FLAGS_deny_cinn_ops"] = "dropout"

            logger.info("run type is {}".format(self.case_name.split("train_")[-1]))
            logger.info("set FLAGS_use_cinn as {}".format(os.getenv("FLAGS_use_cinn")))
            logger.info("set FLAGS_prim_all as {}".format(os.getenv("FLAGS_prim_all")))
        else:
            return 0


def run():
    """
    执行入口
    """
    model = PaddleNLP_Case_Start()
    model.build_prepare()
    return 0


if __name__ == "__main__":
    run()
