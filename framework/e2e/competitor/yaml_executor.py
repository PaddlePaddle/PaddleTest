#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
yaml executor
"""
import sys

from competitor_test.competitive import CompetitorCompareTest
from competitor_test.comptrans import CompeTrans
from competitor_test.tools import STOP_BACKWARD
from utils.yaml_loader import YamlLoader


class RunCase(object):
    """
    run case class
    """

    def __init__(self, file_dir):
        """
        initialize
        """
        self.file_dir = file_dir
        self.yaml = YamlLoader(self.file_dir)

    def get_all_case_name(self):
        """
        get competitor case name
        """
        cases = self.yaml.get_all_case_name()
        # 返回有竞品测试的case_name
        case_list = []
        for case_name in cases:
            case = self.yaml.get_case_info(case_name)
            if case["info"].get("pytorch"):
                case_list.append(case_name)
        return case_list

    def get_docstring(self, case_name):
        """
        get docstring
        """
        case = self.yaml.get_case_info(case_name)
        return case["info"]["desc"]

    def run_case(self, case_name):
        """
        run case
        """
        case = self.yaml.get_case_info(case_name)
        self.exec(case, case_name)

    def exec(self, case, case_name):
        """
        actuator
        """
        tans_obj = CompeTrans(case, 1, 2)
        if tans_obj.stop:
            return
        api = tans_obj.get_function()
        paddle_ins = tans_obj.get_paddle_ins()
        torch_ins = tans_obj.get_torch_ins()
        types = tans_obj.get_dtype()
        torch_place = tans_obj.get_torch_place()
        test_obj = CompetitorCompareTest(*api)
        test_obj.types = types
        if torch_place:
            test_obj.torch_place = True
        if case_name in STOP_BACKWARD:
            test_obj.enable_backward = False
        test_obj.run(paddle_ins, torch_ins)


if __name__ == "__main__":
    obj = RunCase("../yaml/%s.yml" % sys.argv[1])
    r = obj.get_docstring("Tanh")
    print(r)
