# encoding: utf-8
"""
执行case后：回收数据避免占用太多存储空间
"""
import os
import sys
import json
import shutil
import argparse
import yaml
import wget
import numpy as np


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
        print("####", os.listdir(self.reponame))
        return 0

    def build_prepare(self):
        """
        执行准备过程
        """
        # 进入repo中
        ret = 0
        ret = self.remove_data()
        if ret:
            print("build remove_data failed")
            return ret
        return ret


def run():
    """
    执行入口
    """
    model = PaddleClas_End()
    model.build_prepare()
    return 0


if __name__ == "__main__":
    run()
