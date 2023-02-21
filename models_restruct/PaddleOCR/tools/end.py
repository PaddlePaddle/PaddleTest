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
        self.REPO_PATH = os.path.join(os.getcwd(), self.reponame)  # 所有和yaml相关的变量与此拼接
        self.model = os.path.splitext(os.path.basename(self.rd_yaml_path))[0]
        self.category = re.search("/(.*?)/", self.rd_yaml_path).group(1)
        self.TRAIN_LOG_PATH = os.path.join("logs", self.reponame, self.qa_yaml_name, "train_multi.log")
        self.EVAL_LOG_PATH = os.path.join("logs", self.reponame, self.qa_yaml_name, "eval_pretrained.log")

    def getdata1(self, filename, delimiter1, delimiter2):
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
        return data[-1]

    def getdata(filename, kpi):
        """
        get_data
        """
        f = open(filename, encoding="utf-8", errors="ignore")
        for line in f.readlines():
            if kpi + ":" in line:
                regexp = r"%s:(\s*\d+(?:\.\d+)?)" % kpi
                r = re.findall(regexp, line)
                # 如果解析不到符合格式到指标，默认值设置为-1
                kpi_value = float(r[0].strip()) if len(r) > 0 else -1
                print(kpi_value)
        f.close()
        return kpi_value

    def update_json(self, filename, value):
        """
        update_json
        """
        # 1.读取原始json文件
        with open(filename, "r") as f:
            content = json.load(f)
        logger.info("#### content: {}".format(content))

        # 2.更新字典dict
        value_dict = {self.model: value}
        content.update(value_dict)
        logger.info("#### value_dict: {}".format(value_dict))

        # 3.写入
        with open(filename, "w") as f_new:
            json.dump(content, f_new)

    def collect_data_value(self):
        """
        回收之前下载的数据
        """
        # eval acc
        pretrained_yaml_path = os.path.join(os.getcwd(), "tools/ocr_pretrained.yaml")
        pretrained_yaml = yaml.load(open(pretrained_yaml_path, "rb"), Loader=yaml.Loader)
        if self.model in pretrained_yaml[self.category].keys():
            if self.category == "det" or self.category == "kie":
                eval_acc = self.getdata(self.EVAL_LOG_PATH, "hmean")
            elif self.category == "e2e":
                eval_acc = self.getdata(self.EVAL_LOG_PATH, "f_score_e2e")
            elif self.category == "sr":
                eval_acc = self.getdata(self.EVAL_LOG_PATH, "psnr_avg")
            else:
                eval_acc = self.getdata(self.EVAL_LOG_PATH, "acc")
            logger.info("#### eval_acc: {}".format(eval_acc))

        # train loss
        # if self.category == "det":
        #    train_loss = self.getdata(self.TRAIN_LOG_PATH, "loss:", ", loss_shrink_maps")
        # elif self.category == "table":
        #    train_loss = self.getdata(self.TRAIN_LOG_PATH, "loss:", ", horizon_bbox_loss")
        # else:
        #    train_loss = self.getdata(self.TRAIN_LOG_PATH, "loss:", ", avg_reader_cost")

        # parse kpi
        train_loss = self.getdata(self.TRAIN_LOG_PATH, "loss")
        logger.info("#### train_loss: {}".format(train_loss))

        self.update_json("tools/train.json", train_loss)
        self.update_json("tools/eval.json", eval_acc)

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
