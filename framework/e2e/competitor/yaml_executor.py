#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
yaml executor
"""

from competitor_test.competitive import CompetitorCompareTest
from competitor_test.comptrans import CompeTrans
from utils.yaml_loader import YamlLoader


def yaml_exe(yaml_file):
    """
    competitor test run
    """
    obj = YamlLoader(yaml_file)
    cases_name = obj.get_all_case_name()
    for case_name in cases_name:
        case = obj.get_case_info(case_name)
        tans_obj = CompeTrans(case, 1, 2)
        if tans_obj.stop:
            continue
        api = tans_obj.get_function()
        paddle_ins = tans_obj.get_paddle_ins()
        # print(paddle_ins)
        torch_ins = tans_obj.get_torch_ins()
        # print(torch_ins)
        # print(tans_obj.ins)
        types = tans_obj.get_dtype()
        # print(types)
        test_obj = CompetitorCompareTest(*api)
        test_obj.types = types
        test_obj.run(paddle_ins, torch_ins)


if __name__ == "__main__":
    yaml_exe("../yaml/test0.yml")
    # obj = YamlLoader("test0.yml")
    # cases_name = obj.get_all_case_name()
