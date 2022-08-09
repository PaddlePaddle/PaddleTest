#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
lazy runner
"""
import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(os.getcwd())), "utils"))
from utils.yaml_loader import YamlLoader

yaml_path = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())), "utils", "nn.yml")
py_cmd = "python3.8"
# loading yaml
yml = YamlLoader(yaml_path)

cases = yml.get_all_case_name()
print("all cases are here: ", cases)
fail_cases = []

# 执行有bug
# for i in cases:
#     try:
#         jit_case = JitTrans(case=yml.get_case_info(i))
#         jit_case.jit_run()
#     except BaseException as bx:
#         print('case is: ', i)
#         print("lzy异常打印: ", bx)
#         fail_cases.append(i)

for i in cases:
    unit_exit_code = os.system(py_cmd + " unit_runner.py --case " + i)
    print("unit_exit_code is: ", unit_exit_code)
    if unit_exit_code != 0:
        fail_cases.append(i)

print("================ final results ================")
print("fail_cases are: ", fail_cases)
print("fail cases num is: ", len(fail_cases))
if fail_cases is []:
    sys.exit(0)
else:
    sys.exit(1)
