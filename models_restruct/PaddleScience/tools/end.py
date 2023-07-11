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


class PaddleScience_End(object):
    """
    回收类
    """

    def __init__(self):
        """
        初试化
        """
        self.reponame = os.environ["reponame"]
        self.qa_yaml_name = os.environ["qa_yaml_name"]

    def plot_loss(self):
        """
        绘制动转静图片
        """
        path_now = os.getcwd()
        os.chdir("picture")
        try:
            path_list = []
            for name in os.listdir(os.path.join("../logs", self.reponame, self.qa_yaml_name)):
                path_list.append(os.path.join("../logs", self.reponame, self.qa_yaml_name, name))
            plot_loss_from_log_files(path_list, self.qa_yaml_name)
        except Exception as e:
            logger.info("draw picture failed")
            logger.info("error info : {}".format(e))
        os.chdir(path_now)
        return 0

    def build_end(self):
        """
        执行准备过程
        """
        # 进入repo中
        logger.info("build remove_data start")
        ret = 0
        # ret = self.remove_data()
        # if ret:
        #     logger.info("build remove_data failed")
        #     return ret
        mode = os.getenv("mode")
        if mode == "precision":
            url = "https://paddle-qa.bj.bcebos.com/suijiaxin/base_log/{}_base.log".format(self.qa_yaml_name)
            file_name = "{}_base.log".format(self.qa_yaml_name)
            try:
                urllib.request.urlretrieve(url, file_name)
                shutil.copy(file_name, "./logs/{}/{}/".format(self.reponame, self.qa_yaml_name))
            except Exception as e:
                logger.error("Failed to download file from {}. Error: {}".format(url, e))
                raise Exception("File download failed!")
            logger.info("plot start!")
            self.plot_loss()
        logger.info("build remove_data end")
        return ret


def run():
    """
    执行入口
    """
    model = PaddleScience_End()
    model.build_end()
    return 0


if __name__ == "__main__":
    run()
