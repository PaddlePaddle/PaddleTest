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


def test_Hardtanh_base():
    """test Hardtanh_base"""
    jit_case = JitTrans(case=yml.get_case_info("Hardtanh_base"))
    jit_case.jit_run()


def test_Hardtanh_0():
    """test Hardtanh_0"""
    jit_case = JitTrans(case=yml.get_case_info("Hardtanh_0"))
    jit_case.jit_run()


def test_Hardtanh_1():
    """test Hardtanh_1"""
    jit_case = JitTrans(case=yml.get_case_info("Hardtanh_1"))
    jit_case.jit_run()


def test_Hardtanh_2():
    """test Hardtanh_2"""
    jit_case = JitTrans(case=yml.get_case_info("Hardtanh_2"))
    jit_case.jit_run()


def test_Hardtanh_3():
    """test Hardtanh_3"""
    jit_case = JitTrans(case=yml.get_case_info("Hardtanh_3"))
    jit_case.jit_run()


def test_Hardtanh_4():
    """test Hardtanh_4"""
    jit_case = JitTrans(case=yml.get_case_info("Hardtanh_4"))
    jit_case.jit_run()


def test_Hardtanh_5():
    """test Hardtanh_5"""
    jit_case = JitTrans(case=yml.get_case_info("Hardtanh_5"))
    jit_case.jit_run()


def test_Hardtanh_6():
    """test Hardtanh_6"""
    jit_case = JitTrans(case=yml.get_case_info("Hardtanh_6"))
    jit_case.jit_run()


def test_Hardtanh_7():
    """test Hardtanh_7"""
    jit_case = JitTrans(case=yml.get_case_info("Hardtanh_7"))
    jit_case.jit_run()
