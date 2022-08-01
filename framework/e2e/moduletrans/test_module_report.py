#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
yml test
"""
import os
import pytest
import allure
from yaml_loader import YamlLoader
import controller


yaml_path = os.path.join("yaml", "Det", "modeling", "backbones", "blazenet.yml")
yml = YamlLoader(yaml_path)
all_cases_list = ["blazenet_BlazeBlock_0"]

# all_cases_list = []
# all_cases_dict = yml.get_all_case_name()
# for k in all_cases_dict:
#     all_cases_list.append(k)
# print(all_cases_list)


@allure.story("e2e_Layer")
@pytest.mark.parametrize("case_name", all_cases_list)
def test_module_layer(case_name):
    """pytest case"""
    allure.dynamic.title(case_name)
    allure.dynamic.description("Layer 测试")
    case = yml.get_case_info(case_name)
    test = controller.ControlModuleTrans(case=case)
    test.run_test()


case = yml.get_case_info(all_cases_list[0])
test = controller.ControlModuleTrans(case=case)
test.run_test()
