# encoding: utf-8
"""
执行case前：生成不同模型的配置参数，例如算法、字典等
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


class PaddleOCR_Start(object):
    """
    自定义环境准备
    """

    def __init__(self):
        """
        初始化变量
        """
        self.qa_yaml_name = os.environ["qa_yaml_name"]
        self.rd_yaml_path = os.environ["rd_yaml_path"]
        logger.info("###self.qa_yaml_name: {}".format(self.qa_yaml_name))
        self.reponame = os.environ["reponame"]
        self.system = os.environ["system"]
        self.step = os.environ["step"]
        self.paddle_whl = os.environ["paddle_whl"]
        self.mode = os.environ["mode"]  # function or precision
        self.REPO_PATH = os.path.join(os.getcwd(), self.reponame)  # 所有和yaml相关的变量与此拼接
        self.env_dict = {}

    def prepare_config_params(self):
        """
        准备配置参数
        """
        yaml_absolute_path = os.path.join(self.REPO_PATH, self.rd_yaml_path)
        self.rd_config = yaml.load(open(yaml_absolute_path, "rb"), Loader=yaml.Loader)
        algorithm = self.rd_config["Architecture"]["algorithm"]
        self.env_dict["algorithm"] = algorithm

        if "character_dict_path" in self.rd_config.keys():
            rec_dict = self.rd_config["Global"]["character_dict_path"]
            if not rec_dict:
                rec_dict = "ppocr/utils/ic15_dict.txt"
            self.env_dict["rec_dict"] = rec_dict

            image_shape_list = self.rd_config["Eval"]["dataset"]["transforms"][2]["RecResizeImg"]["image_shape"]
            image_shape_list = [str(x) for x in image_shape_list]
            image_shape = ",".join((image_shape_list))
            self.env_dict["image_shape"] = image_shape
            print(image_shape)

    def build_prepare(self):
        """
        执行准备过程
        """
        # 进入repo中
        ret = 0
        ret = self.prepare_config_params()
        if ret:
            logger.info("build prepare_config_params failed")

        os.environ[self.reponame] = json.dumps(self.env_dict)
        return ret


def run():
    """
    执行入口
    """
    model = PaddleOCR_Start()
    model.build_prepare()
    return 0


if __name__ == "__main__":
    run()
