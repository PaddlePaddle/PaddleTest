# encoding: utf-8
"""
执行case前：生成yaml，设置特殊参数，改变监控指标
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


class PaddleClas_Case_Start(object):
    """
    自定义环境准备
    """

    def __init__(self):
        """
        初始化变量
        """
        self.output = "output"
        self.mode = os.environ["mode"]
        self.reponame = os.environ["reponame"]
        self.case_step = os.environ["case_step"]
        self.case_name = os.environ["case_name"]
        self.qa_yaml_name = os.environ["qa_yaml_name"]
        values_str = os.environ.get(self.reponame, "")
        try:
            self.values_dic = json.loads(values_str)
        except:
            self.values_dic = {}

    def prepare_env(self):
        """
        下载预训练模型, 指定路径
        """
        path_now = os.getcwd()  # 切入路径
        os.chdir(self.reponame)
        if self.case_step == "eval":
            self.values_dic["eval_trained_model"] = None  # 赋初始值
            for name in os.listdir(self.output):
                if self.qa_yaml_name.split("^")[-1].split(".yaml")[0] in name:
                    self.values_dic["eval_trained_model"] = os.path.join(
                        self.output, name, "iter_20_checkpoint.pdparams"
                    )

        os.chdir(path_now)  # 切回路径
        return 0

    def build_prepare(self):
        """
        执行准备过程
        """
        ret = self.prepare_env()
        if ret:
            logger.info("build prepare_env failed")
            return ret

        if self.values_dic != {}:
            os.environ[self.reponame] = json.dumps(self.values_dic)
        return ret


def run():
    """
    执行入口
    """
    model = PaddleClas_Case_Start()
    model.build_prepare()
    return 0


if __name__ == "__main__":
    # args = None
    # model = PaddleClas_Case_Start(args)
    # model.build_prepare()
    run()
