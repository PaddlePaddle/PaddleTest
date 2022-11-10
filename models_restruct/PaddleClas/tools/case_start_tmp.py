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
        self.reponame = os.environ["reponame"]
        self.mode = os.environ["mode"]
        self.case_step = os.environ["case_step"]
        self.case_name = os.environ["case_name"]
        self.qa_yaml_name = os.environ["qa_yaml_name"]
        values_str = os.environ.get(self.reponame, "")
        try:
            values_dic = json.loads(values_str)
        except:
            values_dic = {}
        self.kpi_value_eval = values_dic["kpi_value_eval"]

    def change_yaml_kpi(self, content, content_result):
        """
        递归修改变量值,直接全部整体替换,不管现在需要的是什么阶段
        注意这里依赖 self.step 变量, 如果执行时不传入暂时获取不到, TODO:待框架优化
        """
        if isinstance(content, dict):
            for key, val in content.items():
                if isinstance(content[key], dict):
                    self.change_yaml_kpi(content[key], content_result[key])
                elif isinstance(content[key], list):
                    for i, case_value in enumerate(content[key]):
                        for key1, val1 in case_value.items():
                            # print('####key', key)
                            # print('####key1', key1)
                            # print('####self.case_step', self.case_step)
                            if key1 == "result" and key == self.case_step:  # 结果和阶段同时满足
                                content[key][i][key1] = content_result[key][i][key1]
        return content, content_result

    def update_kpi(self):
        """
        根据之前的字典更新kpi监控指标, 原来的数据只起到确定格式, 没有实际用途
        其实可以在这一步把QA需要替换的全局变量给替换了,就不需要框架来做了,重组下qa的yaml
        kpi_name 与框架强相关, 要随框架更新, 目前是支持了单个value替换结果
        """
        # 读取上次执行的产出
        # 通过whl包的地址，判断是release还是develop  report_linux_cuda102_py37_develop

        with open(os.path.join("tools", "report_linux_cuda102_py37_develop.yaml"), "r") as f:
            content_result = yaml.load(f, Loader=yaml.FullLoader)

        if self.qa_yaml_name in content_result.keys():  # 查询yaml中是否存在已获取的模型指标
            with open(os.path.join("cases", self.qa_yaml_name) + ".yaml", "r") as f:
                content = yaml.load(f, Loader=yaml.FullLoader)

            content = json.dumps(content)
            content = content.replace("${{{0}}}".format("kpi_value_eval"), self.kpi_value_eval)
            content = json.loads(content)

            content, content_result = self.change_yaml_kpi(content, content_result[self.qa_yaml_name])

            with open(os.path.join("cases", self.qa_yaml_name) + ".yaml", "w") as f:
                yaml.dump(content, f, sort_keys=False)

    def build_prepare(self):
        """
        执行准备过程
        """
        if self.mode == "precision":
            ret = self.update_kpi()
            if ret:
                logger.info("build update_kpi failed")
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
