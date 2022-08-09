#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
unit runner
"""
import os
import sys
import argparse

sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(os.getcwd())), "utils"))

from utils.yaml_loader import YamlLoader
from jittrans import JitTrans


parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--case", type=str, default=None, help="case for test.")
args = parser.parse_args()

yaml_path = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())), "utils", "nn.yml")
yml = YamlLoader(yaml_path)

if __name__ == "__main__":
    """main"""
    try:
        jit_case = JitTrans(case=yml.get_case_info(args.case))
        jit_case.jit_run()
    except BaseException as bx:
        print("case is: ", args.case)
        print("lzy异常打印: ", bx)
        sys.exit(1)
