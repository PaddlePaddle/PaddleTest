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


class PaddleClas_End(object):
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
            if ".tar" in file_name:
                os.remove(file_name)
                if os.path.exists(file_name.replace(".tar", "")):
                    shutil.rmtree(file_name.replace(".tar", ""))
                logger.info("#### clean data _infer: {}".format(file_name))

            if "_pretrained.pdparams" in file_name:
                os.remove(file_name)
                logger.info("#### clean data: {}".format(file_name))

            if file_name == "inference":
                shutil.rmtree("inference")
                logger.info("#### clean data inference: {}".format("inference"))

            if file_name == "dataset":
                del_dataset = glob.glob(r"dataset/*.tar")
                if del_dataset != []:
                    logger.info("#### clean data dataset: {}".format("dataset"))
                    for del_name in del_dataset:
                        os.remove(del_name)
                if os.path.exists(os.path.join("dataset", "face")):
                    shutil.rmtree(os.path.join("dataset", "face"))
                    logger.info("#### clean data face: {}".format("dataset"))

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
    model = PaddleClas_End()
    model.build_end()
    return 0


if __name__ == "__main__":
    run()
