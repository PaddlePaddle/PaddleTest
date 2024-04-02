#!/bin/env python3
# -*- coding: utf-8 -*-
# @author Zeref996
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
nn.Layer配置相关
"""

import os

# import platform
# import time
# import pytest
# import allure
# from tools.yaml_loader import YamlLoader


class CaseSelect(object):
    """通过指定的nn.Layer的yaml, 选择用于测试的cases"""

    def __init__(self, yaml_dir):
        """init"""
        # self.cur_path = os.getcwd()
        # self.report_dir = os.path.join(self.cur_path, "report")
        # self.env = env
        # self.repo_list = repo_list
        self.yaml_dir = yaml_dir

    def get_yaml_list(self, base_path, yaml_list=[]):
        """递归寻找文件夹内所有的yml文件路径"""
        file_list = os.listdir(base_path)

        for file in file_list:
            yaml_path = os.path.join(base_path, file)

            if os.path.isdir(yaml_path):
                self.get_yaml_list(yaml_path, yaml_list)
            else:
                if not file.endswith(".yml"):
                    continue
                else:
                    yaml_list.append(yaml_path)
        return yaml_list
