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


def test_AdaptiveAvgPool2D_base():
    """test AdaptiveAvgPool2D_base"""
    jit_case = JitTrans(case=yml.get_case_info("AdaptiveAvgPool2D_base"))
    jit_case.jit_run()


def test_AdaptiveAvgPool2D_0():
    """test AdaptiveAvgPool2D_0"""
    jit_case = JitTrans(case=yml.get_case_info("AdaptiveAvgPool2D_0"))
    jit_case.jit_run()


def test_AdaptiveAvgPool2D_1():
    """test AdaptiveAvgPool2D_1"""
    jit_case = JitTrans(case=yml.get_case_info("AdaptiveAvgPool2D_1"))
    jit_case.jit_run()


def test_AdaptiveAvgPool2D_2():
    """test AdaptiveAvgPool2D_2"""
    jit_case = JitTrans(case=yml.get_case_info("AdaptiveAvgPool2D_2"))
    jit_case.jit_run()


def test_AdaptiveAvgPool2D_3():
    """test AdaptiveAvgPool2D_3"""
    jit_case = JitTrans(case=yml.get_case_info("AdaptiveAvgPool2D_3"))
    jit_case.jit_run()


def test_AdaptiveAvgPool2D_4():
    """test AdaptiveAvgPool2D_4"""
    jit_case = JitTrans(case=yml.get_case_info("AdaptiveAvgPool2D_4"))
    jit_case.jit_run()


def test_AdaptiveAvgPool2D_5():
    """test AdaptiveAvgPool2D_5"""
    jit_case = JitTrans(case=yml.get_case_info("AdaptiveAvgPool2D_5"))
    jit_case.jit_run()


def test_AdaptiveAvgPool2D_6():
    """test AdaptiveAvgPool2D_6"""
    jit_case = JitTrans(case=yml.get_case_info("AdaptiveAvgPool2D_6"))
    jit_case.jit_run()


def test_AvgPool2D_base():
    """test AvgPool2D_base"""
    jit_case = JitTrans(case=yml.get_case_info("AvgPool2D_base"))
    jit_case.jit_run()


def test_AvgPool2D_0():
    """test AvgPool2D_0"""
    jit_case = JitTrans(case=yml.get_case_info("AvgPool2D_0"))
    jit_case.jit_run()


def test_AvgPool2D_1():
    """test AvgPool2D_1"""
    jit_case = JitTrans(case=yml.get_case_info("AvgPool2D_1"))
    jit_case.jit_run()


def test_AvgPool2D_2():
    """test AvgPool2D_2"""
    jit_case = JitTrans(case=yml.get_case_info("AvgPool2D_2"))
    jit_case.jit_run()


def test_AvgPool2D_3():
    """test AvgPool2D_3"""
    jit_case = JitTrans(case=yml.get_case_info("AvgPool2D_3"))
    jit_case.jit_run()


def test_AvgPool2D_4():
    """test AvgPool2D_4"""
    jit_case = JitTrans(case=yml.get_case_info("AvgPool2D_4"))
    jit_case.jit_run()


def test_AvgPool2D_5():
    """test AvgPool2D_5"""
    jit_case = JitTrans(case=yml.get_case_info("AvgPool2D_5"))
    jit_case.jit_run()


def test_AvgPool2D_6():
    """test AvgPool2D_6"""
    jit_case = JitTrans(case=yml.get_case_info("AvgPool2D_6"))
    jit_case.jit_run()


def test_AvgPool2D_7():
    """test AvgPool2D_7"""
    jit_case = JitTrans(case=yml.get_case_info("AvgPool2D_7"))
    jit_case.jit_run()
