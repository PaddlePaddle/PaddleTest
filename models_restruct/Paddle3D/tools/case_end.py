# encoding: utf-8
"""
执行case后：获取log中的数值
"""
import os
import sys
import re
import json
import glob
import shutil
import math
import argparse
import logging
import yaml
import wget
import paddle
import allure
import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger("ce")


class Paddle3D_End(object):
    """
    回收类
    """

    def __init__(self):
        """
        初试化
        """
        self.qa_yaml_name = os.environ["qa_yaml_name"]
        self.rd_yaml_path = os.environ["rd_yaml_path"]
        logger.info("#### self.qa_yaml_name: {}".format(self.qa_yaml_name))
        self.reponame = os.environ["reponame"]
        self.system = os.environ["system"]
        self.step = os.environ["step"]
        self.paddle_whl = os.environ["paddle_whl"]
        self.mode = os.environ["mode"]  # function or precision
        self.REPO_PATH = os.path.join(os.getcwd(), self.reponame)  # 所有和yaml相关的变量与此拼接
        self.model = os.path.splitext(os.path.basename(self.rd_yaml_path))[0]
        self.category = re.search("/(.*?)/", self.rd_yaml_path).group(1)


    def build_end(self):
        """
        执行准备过程
        """
        # 进入repo中
        logger.info("This is case_end!")
        logger.info("os.getcwd():{}", os.getcwd())
        os.system("python tools/end.py")
        os.system("python -m pytest -sv test_plot.py  --alluredir=./result")   

def run():
    """
    执行入口
    """
    print("This is Paddle3D_End start!")
    model = Paddle3D_End()
    model.build_end()
    return 0


if __name__ == "__main__":
    run()
