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

    def __init__(self, layer_dir, testing):
        """

        :param layer_dir: 所有layer.yml文件夹路径
        :param testing: 单个testing.yml文件路径
        """
        # 获取所有layer.yml文件路径
        self.layer_dir = layer_dir
        self.py_list = CaseSelect(self.layer_dir).get_py_list(base_path=self.layer_dir)

        self.testing = testing

        self.py_cmd = os.environ.get("PYTHON")
        self.report_dir = os.path.join(os.getcwd(), "report")

    def _test_run(self):
        """run some test"""
        for py_file in self.py_list:
            title = py_file.replace(".py", "").replace("/", "^").replace(".", "^")
            if platform.system() == "Windows":
                # os.system(
                #     "{}.exe -m pytest PaddleLT.py --all_dir"
                #     "={} --yaml={} --case={} --testing={} --alluredir={}".format(
                #         self.py_cmd, self.layer_dir, yaml, case, self.testing, self.report_dir
                #     )
                # )
                pass
            else:
                os.system(
                    "cp -r PaddleLT.py {}.py && "
                    "{} -m pytest {}.py --title={} --layerfile={} --testing={} --alluredir={}".format(
                        title, self.py_cmd, title, title, py_file, self.testing, self.report_dir
                    )
                )

            # for case in YamlLoader(yml=yaml).get_all_case_name():
            #     last_dir = os.path.basename(self.layer_dir)
            #     base_dir = self.layer_dir.replace(last_dir, "")
            #     title = (
            #         yaml.replace(base_dir, "").replace(".yml", ".{}".format(case)).replace("/", "^").replace(".", "^")
            #     )
            #     if platform.system() == "Windows":
            #         os.system(
            #             "{}.exe -m pytest PaddleLT.py --all_dir"
            #             "={} --yaml={} --case={} --testing={} --alluredir={}".format(
            #                 self.py_cmd, self.layer_dir, yaml, case, self.testing, self.report_dir
            #             )
            #         )
            #     else:
            #         os.system(
            #             "cp -r PaddleLT.py {}.py && "
            #             "{} -m pytest {}.py --all_dir={} --yaml={} --case={} --testing={} --alluredir={}".format(
            #                 title, self.py_cmd, title, self.layer_dir, yaml, case, self.testing, self.report_dir
            #             )
            #         )


if __name__ == "__main__":
    test_yml = "yaml/demo_new_testing.yml"
    tes = Run(layer_dir="diy/layer/demo_case", testing=test_yml)
    tes._test_run()
