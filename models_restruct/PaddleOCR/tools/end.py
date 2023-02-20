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
import argparse
import logging
import yaml
import wget
import numpy as np

logger = logging.getLogger("ce")


class PaddleOCR_End(object):
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
        self.LOG_PATH = os.path.join("logs", self.reponame, self.qa_yaml_name, "train_multi.log")
        self.REPO_PATH = os.path.join(os.getcwd(), self.reponame)  # 所有和yaml相关的变量与此拼接
        self.model = os.path.splitext(os.path.basename(self.rd_yaml_path))[0]
        self.category = re.search("/(.*?)/", self.rd_yaml_path).group(1)

    def getdata(self, filename, delimiter1, delimiter2):
        """
        get_data
        """
        data = []
        with open(filename, "rt") as f:
            for line in f:
                pattern = re.compile(delimiter1 + "(.+)" + delimiter2)
                #          pattern=re.compile('loss:(.+), ')
                result = pattern.findall(line)
                if len(result) > 0:
                    # print(float(result[0]))
                    data.append(float(result[0]))
        return np.mean(data)

    def collect_data_value(self):
        """
        回收之前下载的数据
        """
        path_now = os.getcwd()
        os.chdir(self.reponame)
        if self.category == "det":
            loss_data = self.getdata(self.LOG_PATH, "loss:", ", loss_shrink_maps")
        elif self.category == "table":
            loss_data = self.getdata(self.LOG_PATH, "loss:", ", horizon_bbox_loss")
        else:
            loss_data = self.getdata(self.LOG_PATH, "loss:", ", avg_reader_cost")

        # 1.读取原始json文件
        with open(self.LOG_PATH, "r") as f:
            content = json.load(f)

        # 2.更新字典dict
        data_value = {self.model: loss_data}
        content.update(data_value)

        # 3.写入
        with open("tools/train.json", "a") as f_new:
            json.dump(content, f_new)

        os.chdir(path_now)
        return 0

    def build_end(self):
        """
        执行准备过程
        """
        # 进入repo中
        logger.info("build collect data  value start")
        ret = 0
        ret = self.collect_data_value()
        if ret:
            logger.info("build collect_data_value failed")
            return ret
        logger.info("build collect_data_value end")
        return ret


def run():
    """
    执行入口
    """
    model = PaddleOCR_End()
    model.build_end()
    return 0


if __name__ == "__main__":
    run()
