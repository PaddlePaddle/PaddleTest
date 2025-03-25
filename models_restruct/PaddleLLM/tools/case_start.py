# encoding: utf-8
"""
执行case前：生成yaml，设置特殊参数，改变监控指标
"""
import os
import logging
import re

logger = logging.getLogger("ce")


class PaddleLLM_Case_Start(object):
    """
    自定义环境准备
    """

    def __init__(self):
        """
        初始化变量
        """
        self.reponame = os.environ["reponame"]
        self.mode = os.environ["mode"]
        self.case_step = os.environ["case_step"]
        self.case_name = os.environ["case_name"]
        self.qa_yaml_name = os.environ["qa_yaml_name"]

    def build_prepare(self):
        """
        执行准备过程
        """
        logger.info("no prepare")

def run():
    """
    执行入口
    """
    model = PaddleLLM_Case_Start()
    model.build_prepare()


if __name__ == "__main__":
    run()
