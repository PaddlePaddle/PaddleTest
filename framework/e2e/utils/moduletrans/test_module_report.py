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


cur_path = os.getcwd()

if not os.path.exists(os.path.join(cur_path, "ground_truth.tar")):
    os.system("wget https://paddle-qa.bj.bcebos.com/luozeyu01/framework_e2e_LayerTest/ground_truth.tar")
    os.system("tar -xzf ground_truth.tar")

if not os.path.exists(os.path.join(cur_path, "ppcls")):
    os.system("git clone -b develop https://github.com/PaddlePaddle/PaddleClas.git")
    os.system("cd PaddleClas")
    os.system("python -m pip install -r requirements.txt")
    os.system("python setup.py install")
    os.system("cd {}".format(cur_path))

if not os.path.exists(os.path.join(cur_path, "ppdet")):
    os.system("git clone -b develop https://github.com/PaddlePaddle/PaddleDetection.git")
    os.system("cd PaddleDetection")
    os.system("python -m pip install -r requirements.txt")
    os.system("python setup.py install")
    os.system("cd {}".format(cur_path))

yaml_path = "module.yml"
yml = YamlLoader(yaml_path)
all_cases_list = ["Module_10"]

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
