#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
yml test
"""
import os
from yaml_loader import YamlLoader
from logger import logger
import generator.builder

# yaml_path = os.path.join("moduletrans", "module.yml")
yaml_path = "module.yml"
case_name = "Module_3"
yml = YamlLoader(yaml_path)

case = yml.get_case_info(case_name)

print(case)

test = generator.builder.BuildModuleTest(case)
# dygraph_to_static_train_test = test.dygraph_to_static_train_test()
# print("dygraph_to_static_train_test is: ", dygraph_to_static_train_test)


dygraph_to_static_predict_test = test.dygraph_to_static_predict_test()
print("dygraph_to_static_predict_test is: ", dygraph_to_static_predict_test)
