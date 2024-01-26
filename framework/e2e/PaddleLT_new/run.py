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

    def __init__(self):
        """
        init
        """
        # 获取所有layer.yml文件路径
        self.layer_dir = os.env.get("CASE_DIR")
        self.py_list = CaseSelect(self.layer_dir).get_py_list(base_path=self.layer_dir)

        self.testing = os.env.get("TESTING")

        self.py_cmd = os.environ.get("python_ver")
        self.report_dir = os.path.join(os.getcwd(), "report")

    def _test_run(self):
        """run some test"""
        for py_file in self.py_list:
            title = py_file.replace(".py", "").replace("/", "^").replace(".", "^")
            if platform.system() == "Windows":
                pass
            else:
                os.system(
                    "cp -r PaddleLT.py {}.py && "
                    "{} -m pytest {}.py --title={} --layerfile={} --testing={} --alluredir={}".format(
                        title, self.py_cmd, title, title, py_file, self.testing, self.report_dir
                    )
                )


if __name__ == "__main__":
    tes = Run()
    tes._test_run()
