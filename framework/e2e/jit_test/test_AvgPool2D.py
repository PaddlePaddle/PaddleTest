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


def test_AdaptiveAvgPool2D():
    """test AdaptiveAvgPool2D"""
    jit_case = JitTrans(case=yml.get_case_info("AdaptiveAvgPool2D"))
    jit_case.jit_run()


def test_AdaptiveAvgPool2D0():
    """test AdaptiveAvgPool2D0"""
    jit_case = JitTrans(case=yml.get_case_info("AdaptiveAvgPool2D0"))
    jit_case.jit_run()


def test_AdaptiveAvgPool2D1():
    """test AdaptiveAvgPool2D1"""
    jit_case = JitTrans(case=yml.get_case_info("AdaptiveAvgPool2D1"))
    jit_case.jit_run()


def test_AdaptiveAvgPool2D2():
    """test AdaptiveAvgPool2D2"""
    jit_case = JitTrans(case=yml.get_case_info("AdaptiveAvgPool2D2"))
    jit_case.jit_run()


def test_AdaptiveAvgPool2D3():
    """test AdaptiveAvgPool2D3"""
    jit_case = JitTrans(case=yml.get_case_info("AdaptiveAvgPool2D3"))
    jit_case.jit_run()


def test_AdaptiveAvgPool2D4():
    """test AdaptiveAvgPool2D4"""
    jit_case = JitTrans(case=yml.get_case_info("AdaptiveAvgPool2D4"))
    jit_case.jit_run()


def test_AdaptiveAvgPool2D5():
    """test AdaptiveAvgPool2D5"""
    jit_case = JitTrans(case=yml.get_case_info("AdaptiveAvgPool2D5"))
    jit_case.jit_run()


def test_AvgPool2D_base():
    """test AvgPool2D_base"""
    jit_case = JitTrans(case=yml.get_case_info("AvgPool2D_base"))
    jit_case.jit_run()


def test_AvgPool2D():
    """test AvgPool2D"""
    jit_case = JitTrans(case=yml.get_case_info("AvgPool2D"))
    jit_case.jit_run()


def test_AvgPool2D0():
    """test AvgPool2D0"""
    jit_case = JitTrans(case=yml.get_case_info("AvgPool2D0"))
    jit_case.jit_run()


def test_AvgPool2D1():
    """test AvgPool2D1"""
    jit_case = JitTrans(case=yml.get_case_info("AvgPool2D1"))
    jit_case.jit_run()


def test_AvgPool2D2():
    """test AvgPool2D2"""
    jit_case = JitTrans(case=yml.get_case_info("AvgPool2D2"))
    jit_case.jit_run()


def test_AvgPool2D3():
    """test AvgPool2D3"""
    jit_case = JitTrans(case=yml.get_case_info("AvgPool2D3"))
    jit_case.jit_run()


def test_AvgPool2D4():
    """test AvgPool2D4"""
    jit_case = JitTrans(case=yml.get_case_info("AvgPool2D4"))
    jit_case.jit_run()


def test_AvgPool2D5():
    """test AvgPool2D5"""
    jit_case = JitTrans(case=yml.get_case_info("AvgPool2D5"))
    jit_case.jit_run()


def test_AvgPool2D6():
    """test AvgPool2D6"""
    jit_case = JitTrans(case=yml.get_case_info("AvgPool2D6"))
    jit_case.jit_run()
