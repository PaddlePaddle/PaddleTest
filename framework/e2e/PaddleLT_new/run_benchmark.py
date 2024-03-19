#!/bin/env python3
# -*- coding: utf-8 -*-
# @author Zeref996
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
测试执行器
"""
import os
import platform
import layertest
import pandas as pd
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
        # self.layer_dir = os.environ.get("CASE_DIR")
        self.layer_dir = os.path.join("layercase", os.environ.get("CASE_DIR"))

        # 获取需要忽略的case
        self.ignore_list = YamlLoader(yml=os.path.join("yaml", "ignore_case.yml")).yml.get(os.environ.get("CASE_DIR"))

        # 获取待测case
        self.py_list = CaseSelect(self.layer_dir, self.ignore_list).get_py_list(base_path=self.layer_dir)

        self.testing = os.environ.get("TESTING")
        self.py_cmd = os.environ.get("python_ver")
        self.report_dir = os.path.join(os.getcwd(), "report")

    def _test_run(self):
        """run some test"""
        sublayer_dict = {}
        error_count = 0
        error_list = []
        for py_file in self.py_list:
            title = py_file.replace(".py", "").replace("/", "^").replace(".", "^")
            single_test = layertest.LayerTest(title=title, layerfile=py_file, testing=self.testing)
            perf_dict, exit_code = single_test._perf_case_run()
            # title = py_file.replace(".py", "").replace("/", "^").replace(".", "^")
            # exit_code = os.system(
            #     "cp -r PaddleLT.py {}.py && "
            #     "{} -m pytest {}.py --title={} --layerfile={} --testing={} --alluredir={}".format(
            #         title, self.py_cmd, title, title, py_file, self.testing, self.report_dir
            #     )
            # )

            # 报错的子图+engine将不会收录进sublayer_dict
            if exit_code != 0:
                error_list.append(py_file)
                error_count += 1
                continue

            sublayer_dict[title] = perf_dict

        if error_count != 0:
            print("测试失败，报错子图为: {}".format(error_list))
            os.system("echo 7 > exit_code.txt")
        else:
            print("测试通过，无报错子图-。-")
            os.system("echo 0 > exit_code.txt")

        data = [
            {"Key": key, "Value": value}
            for key, sublayer_dict in sublayer_dict.items()
            for value in sublayer_dict.values()
        ]
        # 创建 DataFrame
        df = pd.DataFrame(data)

        # 将数据写入 Excel 文件
        excel_file = "output.xlsx"  # 输出的 Excel 文件名
        df.to_excel(excel_file, index=False)


if __name__ == "__main__":
    tes = Run()
    tes._test_run()
