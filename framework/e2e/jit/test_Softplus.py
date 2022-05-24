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


def test_Softplus_base():
    """test Softplus_base"""
    jit_case = JitTrans(case=yml.get_case_info("Softplus_base"))
    jit_case.jit_run()


def test_Softplus_0():
    """test Softplus_0"""
    jit_case = JitTrans(case=yml.get_case_info("Softplus_0"))
    jit_case.jit_run()


def test_Softplus_1():
    """test Softplus_1"""
    jit_case = JitTrans(case=yml.get_case_info("Softplus_1"))
    jit_case.jit_run()


def test_Softplus_2():
    """test Softplus_2"""
    jit_case = JitTrans(case=yml.get_case_info("Softplus_2"))
    jit_case.jit_run()


def test_Softplus_3():
    """test Softplus_3"""
    jit_case = JitTrans(case=yml.get_case_info("Softplus_3"))
    jit_case.jit_run()


def test_Softplus_4():
    """test Softplus_4"""
    jit_case = JitTrans(case=yml.get_case_info("Softplus_4"))
    jit_case.jit_run()


def test_Softplus_5():
    """test Softplus_5"""
    jit_case = JitTrans(case=yml.get_case_info("Softplus_5"))
    jit_case.jit_run()


def test_Softplus_6():
    """test Softplus_6"""
    jit_case = JitTrans(case=yml.get_case_info("Softplus_6"))
    jit_case.jit_run()
