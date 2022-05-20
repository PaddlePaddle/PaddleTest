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


def test_AdaptiveAvgPool3D():
    """test AdaptiveAvgPool3D"""
    jit_case = JitTrans(case=yml.get_case_info("AdaptiveAvgPool3D"))
    jit_case.jit_run()


def test_AdaptiveAvgPool3D0():
    """test AdaptiveAvgPool3D0"""
    jit_case = JitTrans(case=yml.get_case_info("AdaptiveAvgPool3D0"))
    jit_case.jit_run()


def test_AdaptiveAvgPool3D1():
    """test AdaptiveAvgPool3D1"""
    jit_case = JitTrans(case=yml.get_case_info("AdaptiveAvgPool3D1"))
    jit_case.jit_run()


def test_AdaptiveAvgPool3D2():
    """test AdaptiveAvgPool3D2"""
    jit_case = JitTrans(case=yml.get_case_info("AdaptiveAvgPool3D2"))
    jit_case.jit_run()


def test_AdaptiveAvgPool3D3():
    """test AdaptiveAvgPool3D3"""
    jit_case = JitTrans(case=yml.get_case_info("AdaptiveAvgPool3D3"))
    jit_case.jit_run()


def test_AdaptiveAvgPool3D4():
    """test AdaptiveAvgPool3D4"""
    jit_case = JitTrans(case=yml.get_case_info("AdaptiveAvgPool3D4"))
    jit_case.jit_run()


def test_AdaptiveAvgPool3D5():
    """test AdaptiveAvgPool3D5"""
    jit_case = JitTrans(case=yml.get_case_info("AdaptiveAvgPool3D5"))
    jit_case.jit_run()


def test_AdaptiveAvgPool3D6():
    """test AdaptiveAvgPool3D6"""
    jit_case = JitTrans(case=yml.get_case_info("AdaptiveAvgPool3D6"))
    jit_case.jit_run()


def test_AdaptiveAvgPool3D7():
    """test AdaptiveAvgPool3D7"""
    jit_case = JitTrans(case=yml.get_case_info("AdaptiveAvgPool3D7"))
    jit_case.jit_run()
