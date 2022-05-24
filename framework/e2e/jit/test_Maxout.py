#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test jit cases
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(os.getcwd())), "utils"))

from utils.yaml_loader import YamlLoader
from jittrans import JitTrans

yaml_path = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())), "yaml", "nn.yml")
yml = YamlLoader(yaml_path)


def test_Maxout_base():
    """test Maxout_base"""
    jit_case = JitTrans(case=yml.get_case_info("Maxout_base"))
    jit_case.jit_run()


def test_Maxout_0():
    """test Maxout_0"""
    jit_case = JitTrans(case=yml.get_case_info("Maxout_0"))
    jit_case.jit_run()


def test_Maxout_1():
    """test Maxout_1"""
    jit_case = JitTrans(case=yml.get_case_info("Maxout_1"))
    jit_case.jit_run()


def test_Maxout_2():
    """test Maxout_2"""
    jit_case = JitTrans(case=yml.get_case_info("Maxout_2"))
    jit_case.jit_run()
