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


def test_Dropout_base():
    """test Dropout_base"""
    jit_case = JitTrans(case=yml.get_case_info("Dropout_base"))
    jit_case.jit_run()


def test_Dropout():
    """test Dropout"""
    jit_case = JitTrans(case=yml.get_case_info("Dropout"))
    jit_case.jit_run()


def test_Dropout0():
    """test Dropout0"""
    jit_case = JitTrans(case=yml.get_case_info("Dropout0"))
    jit_case.jit_run()


def test_Dropout1():
    """test Dropout1"""
    jit_case = JitTrans(case=yml.get_case_info("Dropout1"))
    jit_case.jit_run()


def test_Dropout2():
    """test Dropout2"""
    jit_case = JitTrans(case=yml.get_case_info("Dropout2"))
    jit_case.jit_run()


def test_Dropout3():
    """test Dropout3"""
    jit_case = JitTrans(case=yml.get_case_info("Dropout3"))
    jit_case.jit_run()
