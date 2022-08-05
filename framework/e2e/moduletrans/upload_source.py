#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
upload source
"""

import os
from yaml_loader import YamlLoader
import controller


source = "ground_truth"
env = "Mac"
yaml_path = "module.yml"
yml = YamlLoader(yaml_path)
# all_cases_list = ["Module_10"]


all_cases_list = []
all_cases_dict = yml.get_all_case_name()
for k in all_cases_dict:
    all_cases_list.append(k)
print(all_cases_list)


os.system("wget -q --no-proxy https://xly-devops.bj.bcebos.com/home/bos_new.tar.gz")
os.system("tar -xzf bos_new.tar.gz")


# 上传真值ground_truth
def upload_source(cases_list=all_cases_list):
    """upload_source"""
    for case_name in cases_list:
        case = yml.get_case_info(case_name)
        test = controller.ControlModuleTrans(case=case)
        test.mk_dygraph_train_ground_truth()
        test.mk_dygraph_predict_ground_truth()
        test.mk_static_train_ground_truth()
        test.mk_static_predict_ground_truth()


upload_source(all_cases_list)
os.system("tar -czf {}.tar {}".format(source, source))
os.system(
    "python BosClient.py {}.tar paddle-qa/luozeyu01/framework_e2e_LayerTest/{} "
    "https://paddle-qa.bj.bcebos.com/luozeyu01/framework_e2e_LayerTest/{}".format(source, env, env)
)
