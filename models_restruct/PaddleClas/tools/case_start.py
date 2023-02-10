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


class PaddleClas_Case_Start(object):
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
        if "CINN_convergence" in self.qa_yaml_name:
            logger.info("CINN_convergence tag is: {}".format(self.case_name.split("train_")[-1]))
            
            os.environ["FLAGS_prim_all"] = ""
            logger.info("set FLAGS_prim_all as {}".format(os.getenv("FLAGS_prim_all")))
            os.environ["FLAGS_use_cinn"] = ""
            logger.info("set FLAGS_use_cinn as {}".format(os.getenv("FLAGS_use_cinn")))
            os.environ["FLAGS_CONVERT_GRAPH_TO_PROGRAM"] = "1"
            logger.info("set FLAGS_CONVERT_GRAPH_TO_PROGRAM {}".format(os.getenv("FLAGS_CONVERT_GRAPH_TO_PROGRAM")))
            os.environ["NVIDIA_TF32_OVERRIDE"] = "1"
            logger.info("set NVIDIA_TF32_OVERRIDE as {}".format(os.getenv("NVIDIA_TF32_OVERRIDE")))
            os.environ["FLAGS_cudnn_exhaustive_search"] = "1"
            logger.info("set FLAGS_cudnn_exhaustive_search as {}".format(os.getenv("FLAGS_cudnn_exhaustive_search")))
            os.environ["FLAGS_conv_workspace_size_limit"] = "400"
            logger.info("set FLAGS_conv_workspace_size_limit {}".format(os.getenv("FLAGS_conv_workspace_size_limit")))

            if "dy2st_cinn" == self.case_name.split("train_")[-1]:
                os.environ["FLAGS_use_cinn"] = "1"
                logger.info("set FLAGS_use_cinn as {}".format(os.getenv("FLAGS_use_cinn")))
            elif "dy2st_prim" == self.case_name.split("train_")[-1]:
                os.environ["FLAGS_prim_all"] = "true"
                logger.info("set FLAGS_prim_all as {}".format(os.getenv("FLAGS_prim_all")))
            elif "dy2st_prim_cinn" == self.case_name.split("train_")[-1]:
                os.environ["FLAGS_use_cinn"] = "1"
                os.environ["FLAGS_prim_all"] = "true"
                logger.info("set FLAGS_use_cinn as {}".format(os.getenv("FLAGS_use_cinn")))
                logger.info("set FLAGS_prim_all as {}".format(os.getenv("FLAGS_prim_all")))
            else:
                logger.info("use defalt FLAGS_use_cinn is {}".format(os.getenv("FLAGS_use_cinn")))
                logger.info("use defalt FLAGS_prim_all is {}".format(os.getenv("FLAGS_prim_all")))
        else:
            return 0


def run():
    """
    执行入口
    """
    model = PaddleClas_Case_Start()
    model.build_prepare()
    return 0


if __name__ == "__main__":
    # args = None
    # model = PaddleClas_Case_Start(args)
    # model.build_prepare()
    run()
