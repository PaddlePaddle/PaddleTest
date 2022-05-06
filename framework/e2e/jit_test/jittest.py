#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
main_test
"""
import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(os.getcwd())), "utils"))
from utils.yaml_loader import YamlLoader
from jittrans import JitTrans

yaml_path = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())), "utils", "base.yml")

case_name = "digamma"


# loading yaml
# def test():
#     """
#     unit test func
#     """
#     yml = YamlLoader(yaml_path)
#     jit_case = JitTrans(yml.get_case_info("GRUCell"), logger=yml.logger)
#     jit_case.jit_run()


yml = YamlLoader(yaml_path)
jit_case = JitTrans(yml.get_case_info(case_name))
jit_case.jit_run()
