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


def test_Softplus():
    """test Softplus"""
    jit_case = JitTrans(case=yml.get_case_info("Softplus"))
    jit_case.jit_run()


def test_Softplus0():
    """test Softplus0"""
    jit_case = JitTrans(case=yml.get_case_info("Softplus0"))
    jit_case.jit_run()


def test_Softplus1():
    """test Softplus1"""
    jit_case = JitTrans(case=yml.get_case_info("Softplus1"))
    jit_case.jit_run()


def test_Softplus2():
    """test Softplus2"""
    jit_case = JitTrans(case=yml.get_case_info("Softplus2"))
    jit_case.jit_run()


def test_Softplus3():
    """test Softplus3"""
    jit_case = JitTrans(case=yml.get_case_info("Softplus3"))
    jit_case.jit_run()


def test_Softplus4():
    """test Softplus4"""
    jit_case = JitTrans(case=yml.get_case_info("Softplus4"))
    jit_case.jit_run()


def test_Softplus5():
    """test Softplus5"""
    jit_case = JitTrans(case=yml.get_case_info("Softplus5"))
    jit_case.jit_run()
