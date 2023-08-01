# encoding: utf-8
"""
执行case后：回收数据避免占用太多存储空间
"""
import os
import sys
import json
import glob
import shutil
import argparse
import logging
import urllib.request
import yaml
import wget
import numpy as np

# from picture.analysis import analysis, draw
from picture.analysis import plot_loss_from_log_files

logger = logging.getLogger("ce")


class DeepXDE_End(object):
    """
    回收类
    """

    def __init__(self):
        """
        初试化
        """
        self.reponame = os.environ["reponame"]
        self.qa_yaml_name = os.environ["qa_yaml_name"]


def run():
    """
    执行入口
    """
    # model = DeepXDE_End()
    return 0


if __name__ == "__main__":
    run()
