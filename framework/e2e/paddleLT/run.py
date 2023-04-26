#!/bin/env python3
# -*- coding: utf-8 -*-
# @author Zeref996
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
测试执行器
"""
import os
import platform
from tools.case_select import CaseSelect
from tools.yaml_loader import YamlLoader


class Run(object):
    """
    最终执行接口
    """

    def __init__(self, yaml_dir, testing):
        """

        :param yaml_dir: 所有layer.yml文件夹路径
        :param testing: 单个testing.yml文件路径
        """
        # 获取所有layer.yml文件路径
        self.yaml_dir = yaml_dir
        self.yaml_list = CaseSelect(self.yaml_dir).get_yaml_list(base_path=self.yaml_dir)

        self.testing = testing

        self.py_cmd = "python"
        self.report_dir = os.path.join(os.getcwd(), "report")

    def _test_run(self):
        """run some test"""
        for yaml in self.yaml_list:
            for case in YamlLoader(yml=yaml).get_all_case_name():
                if platform.system() == "Windows":
                    os.system(
                        "{}.exe -m pytest PaddleLT.py --all_dir"
                        "={} --yaml={} --case={} --testing={} --alluredir={}".format(
                            self.py_cmd, self.yaml_dir, yaml, case, self.testing, self.report_dir
                        )
                    )
                else:
                    os.system(
                        "{} -m pytest PaddleLT.py --all_dir={} --yaml={} --case={} --testing={} --alluredir={}".format(
                            self.py_cmd, self.yaml_dir, yaml, case, self.testing, self.report_dir
                        )
                    )


if __name__ == "__main__":
    test_yml = "yaml/dy2st_train.yml"
    tes = Run(yaml_dir="yaml/demo_yml", testing=test_yml)
    tes._test_run()
