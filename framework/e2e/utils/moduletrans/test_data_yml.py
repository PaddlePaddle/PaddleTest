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
import controller


# yaml_path = os.path.join("moduletrans", "module.yml")
yaml_path = "Data_test.yml"
case_name = "Data_base"
yml = YamlLoader(yaml_path)

case = yml.get_case_info(case_name)

# def test_module_layer():
#     """pytest case"""
#     test = controller.ControlModuleTrans(case=case)
#     test.run_test()


test = controller.ControlModuleTrans(case=case)
test.run_test()
