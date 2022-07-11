#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
yaml executor
"""
import sys
import pytest

from competitor_test.competitive import CompetitorCompareTest
from competitor_test.comptrans import CompeTrans
from competitor_test.tools import STOP_BACKWARD
from utils.yaml_loader import YamlLoader


def run_case(case, case_name):
    """
    run case
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


def debug_yaml_exe(yaml_file, case_name):
    """
    competitor test run
    """
    obj = YamlLoader(yaml_file)
    case = obj.get_case_info(case_name)

    tans_obj = CompeTrans(case, 1, 2)
    if not tans_obj.stop:
        api = tans_obj.get_function()
        paddle_ins = tans_obj.get_paddle_ins()
        # print(paddle_ins)
        torch_ins = tans_obj.get_torch_ins()
        # print(torch_ins)
        # print(tans_obj.ins)
        types = tans_obj.get_dtype()
        # print(types)
        torch_place = tans_obj.get_torch_place()
        test_obj = CompetitorCompareTest(*api)
        test_obj.types = types
        if torch_place:
            test_obj.torch_place = True
        if case_name in STOP_BACKWARD:
            test_obj.enable_backward = False
        test_obj.run(paddle_ins, torch_ins)


def generate_case_info(yaml_file):
    """
    generate case info
    """
    obj = YamlLoader(yaml_file)
    cases_name = obj.get_all_case_name()
    case_info = []
    for case_name in cases_name:
        case = obj.get_case_info(case_name)
        if case["info"].get("pytorch"):
            case_info.append([case, case_name])
    return case_info


@pytest.fixture()
def ecp(request):
    """
    before run case
    """
    x = request.param
    print("<<<!!! %s !!!>>>" % x[1])
    return x


@pytest.mark.parametrize("ecp", generate_case_info("../yaml/%s.yml" % sys.argv[1]), indirect=True)
def test(ecp):
    """
    run case
    """
    run_case(*ecp)


if __name__ == "__main__":
    # yaml_exe("../yaml/%s.yml" % sys.argv[1])
    # debug_yaml_exe("../yaml/base.yml", sys.argv[1])
    pytest.main(["-sv", "--alluredir=report", sys.argv[0]])
