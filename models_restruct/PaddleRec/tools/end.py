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
import yaml
import wget
import numpy as np

logger = logging.getLogger("ce")


class PaddleRec_End(object):
    """
    回收类
    """

    def __init__(self):
        """
        初试化
        """
        self.reponame = os.environ["reponame"]

    def remove_data(self):
        """
        回收之前下载的数据
        """
        list_dir = os.listdir(self.reponame)
        path_now = os.getcwd()
        os.chdir(self.reponame)

        for file_name in list_dir:
            if "output" in file_name:
                shutil.rmtree(file_name)
                logger.info("#### clean data: {}".format(file_name))

        os.chdir(path_now)
        return 0

    def build_end(self):
        """
        执行准备过程
        """
        # 进入repo中
        logger.info("build remove_data start")
        ret = 0
        ret = self.remove_data()
        if ret:
            logger.info("build remove_data failed")
            return ret
        logger.info("build remove_data end")
        return ret


def run():
    """
    执行入口
    """
    model = PaddleRec_End()
    model.build_end()
    return 0


if __name__ == "__main__":
    run()
