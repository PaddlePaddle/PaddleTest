#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
yaml base
"""

import yaml

# from old_design.logger import Logger, logger


class YamlLoader(object):
    """
    yaml_loader
    """

    def __init__(self, yml):
        """initialize"""
        try:
            with open(yml, encoding="utf-8") as f:
                self.yml = yaml.load(f, Loader=yaml.FullLoader)
        except Exception as e:
            print(e)
        # self.logger = logger

    def __str__(self):
        """str"""
        return str(self.yml)

    def get_case_info(self, case_name):
        """
        get case info
        """
        # self.logger.get_log().info("get ->{}<- case profile".format(case_name))
        return {"info": self.yml.get(case_name), "name": case_name}

    def get_all_case_name(self):
        """
        get all case name
        """
        # 获取全部case name
        return self.yml.keys()
