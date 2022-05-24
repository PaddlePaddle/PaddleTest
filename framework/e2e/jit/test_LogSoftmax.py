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


def test_LogSoftmax_base():
    """test LogSoftmax_base"""
    jit_case = JitTrans(case=yml.get_case_info("LogSoftmax_base"))
    jit_case.jit_run()


def test_LogSoftmax_0():
    """test LogSoftmax_0"""
    jit_case = JitTrans(case=yml.get_case_info("LogSoftmax_0"))
    jit_case.jit_run()


def test_LogSoftmax_1():
    """test LogSoftmax_1"""
    jit_case = JitTrans(case=yml.get_case_info("LogSoftmax_1"))
    jit_case.jit_run()


def test_LogSoftmax_2():
    """test LogSoftmax_2"""
    jit_case = JitTrans(case=yml.get_case_info("LogSoftmax_2"))
    jit_case.jit_run()


def test_LogSoftmax_3():
    """test LogSoftmax_3"""
    jit_case = JitTrans(case=yml.get_case_info("LogSoftmax_3"))
    jit_case.jit_run()


def test_LogSoftmax_4():
    """test LogSoftmax_4"""
    jit_case = JitTrans(case=yml.get_case_info("LogSoftmax_4"))
    jit_case.jit_run()


def test_LogSoftmax_5():
    """test LogSoftmax_5"""
    jit_case = JitTrans(case=yml.get_case_info("LogSoftmax_5"))
    jit_case.jit_run()
