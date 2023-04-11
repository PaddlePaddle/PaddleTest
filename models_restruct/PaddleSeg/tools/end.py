# encoding: utf-8
"""
执行case后执行的文件
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
import matplotlib.pyplot as plt

logger = logging.getLogger("ce")


class PaddleDetection_End(object):
    """
    case执行结束后
    """

    def __init__(self):
        """
        初始化
        """
        self.reponame = os.environ["reponame"]
        self.model = os.environ["model"]
        self.qa_model_name = os.environ["qa_yaml_name"]
        self.log_dir = "logs"
        self.log_name = "train_prim_single.log"
        self.log_path = os.path.join(os.getcwd(), self.log_dir, self.reponame, self.qa_model_name, self.log_name)
        logger.info("log_path:{}".format(self.log_path))
        self.qa_model_name_base = ""
        self.prim_log_path = ""
        self.standard_log_path = ""
        if "prim" in self.qa_model_name:
            self.qa_model_name_base = self.qa_model_name.replace("prim", "static")
            self.prim_log_path = self.log_path
            self.standard_log_path = os.path.join(
                os.getcwd(), self.log_dir, self.reponame, self.qa_model_name_base, self.log_name
            )

    def get_loss(self, log_path):
        """
        获取loss值
        """
        step = []
        loss = []
        num = 0
        fl = open(log_path, "r").readlines()
        for row in fl:
            if "epoch:" in row.strip():
                member = row.strip().split(",")
                for item in member:
                    if "loss" in item:
                        print("item:{}".format(item))
                        loss_item = item.strip().split(":")[-1]
                        print("loss_item:{}".format(loss_item))
                        loss.append(float(loss_item.strip()))
                        step.append(num)
                        num += 1
        return step, loss

    def draw_curve(self):
        """
        绘制曲线
        """
        logger.info("***draw curve start")
        step1, loss1 = self.get_loss(self.standard_log_path)
        step2, loss2 = self.get_loss(self.prim_log_path)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        ax.plot(step1, loss1, label="train_base", color="b", linewidth=2)
        ax.plot(step2, loss2, label="train_prim", color="r", linewidth=2)
        ax.legend()
        ax.set_xlabel("steps")
        ax.set_ylabel("loss")
        ax.set_title("model")
        plt.savefig("model_curve_pic.png")
        os.system("mkdir picture")
        os.system("mv model_curve_pic.png picture")
        return 0

    def build_end(self):
        """
        执行准备过程
        """
        ret = 0
        if "prim" in self.qa_model_name:
            ret = self.draw_curve()
            if ret:
                logger.info("draw curve failed!")
                return ret
            logger.info("draw curve end!")
        return ret


def run():
    """
    执行入口
    """
    model = PaddleDetection_End()
    model.build_end()
    return 0


if __name__ == "__main__":
    run()
