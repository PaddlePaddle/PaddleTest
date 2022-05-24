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


def test_AdaptiveAvgPool3D_base():
    """test AdaptiveAvgPool3D_base"""
    jit_case = JitTrans(case=yml.get_case_info("AdaptiveAvgPool3D_base"))
    jit_case.jit_run()


def test_AdaptiveAvgPool3D_0():
    """test AdaptiveAvgPool3D_0"""
    jit_case = JitTrans(case=yml.get_case_info("AdaptiveAvgPool3D_0"))
    jit_case.jit_run()


def test_AdaptiveAvgPool3D_1():
    """test AdaptiveAvgPool3D_1"""
    jit_case = JitTrans(case=yml.get_case_info("AdaptiveAvgPool3D_1"))
    jit_case.jit_run()


def test_AdaptiveAvgPool3D_2():
    """test AdaptiveAvgPool3D_2"""
    jit_case = JitTrans(case=yml.get_case_info("AdaptiveAvgPool3D_2"))
    jit_case.jit_run()


def test_AdaptiveAvgPool3D_3():
    """test AdaptiveAvgPool3D_3"""
    jit_case = JitTrans(case=yml.get_case_info("AdaptiveAvgPool3D_3"))
    jit_case.jit_run()


def test_AdaptiveAvgPool3D_4():
    """test AdaptiveAvgPool3D_4"""
    jit_case = JitTrans(case=yml.get_case_info("AdaptiveAvgPool3D_4"))
    jit_case.jit_run()


def test_AdaptiveAvgPool3D_5():
    """test AdaptiveAvgPool3D_5"""
    jit_case = JitTrans(case=yml.get_case_info("AdaptiveAvgPool3D_5"))
    jit_case.jit_run()


def test_AdaptiveAvgPool3D_6():
    """test AdaptiveAvgPool3D_6"""
    jit_case = JitTrans(case=yml.get_case_info("AdaptiveAvgPool3D_6"))
    jit_case.jit_run()


def test_AdaptiveAvgPool3D_7():
    """test AdaptiveAvgPool3D_7"""
    jit_case = JitTrans(case=yml.get_case_info("AdaptiveAvgPool3D_7"))
    jit_case.jit_run()


def test_AdaptiveAvgPool3D_8():
    """test AdaptiveAvgPool3D_8"""
    jit_case = JitTrans(case=yml.get_case_info("AdaptiveAvgPool3D_8"))
    jit_case.jit_run()
