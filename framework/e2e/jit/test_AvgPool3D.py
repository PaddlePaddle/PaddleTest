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


def test_AvgPool3D_base():
    """test AvgPool3D_base"""
    jit_case = JitTrans(case=yml.get_case_info("AvgPool3D_base"))
    jit_case.jit_run()


def test_AvgPool3D():
    """test AvgPool3D"""
    jit_case = JitTrans(case=yml.get_case_info("AvgPool3D"))
    jit_case.jit_run()


def test_AvgPool3D0():
    """test AvgPool3D0"""
    jit_case = JitTrans(case=yml.get_case_info("AvgPool3D0"))
    jit_case.jit_run()


def test_AvgPool3D1():
    """test AvgPool3D1"""
    jit_case = JitTrans(case=yml.get_case_info("AvgPool3D1"))
    jit_case.jit_run()


def test_AvgPool3D2():
    """test AvgPool3D2"""
    jit_case = JitTrans(case=yml.get_case_info("AvgPool3D2"))
    jit_case.jit_run()


def test_AvgPool3D3():
    """test AvgPool3D3"""
    jit_case = JitTrans(case=yml.get_case_info("AvgPool3D3"))
    jit_case.jit_run()


def test_AvgPool3D4():
    """test AvgPool3D4"""
    jit_case = JitTrans(case=yml.get_case_info("AvgPool3D4"))
    jit_case.jit_run()


def test_AvgPool3D5():
    """test AvgPool3D5"""
    jit_case = JitTrans(case=yml.get_case_info("AvgPool3D5"))
    jit_case.jit_run()


def test_AvgPool3D6():
    """test AvgPool3D6"""
    jit_case = JitTrans(case=yml.get_case_info("AvgPool3D6"))
    jit_case.jit_run()


def test_AvgPool3D7():
    """test AvgPool3D7"""
    jit_case = JitTrans(case=yml.get_case_info("AvgPool3D7"))
    jit_case.jit_run()


def test_AvgPool3D8():
    """test AvgPool3D8"""
    jit_case = JitTrans(case=yml.get_case_info("AvgPool3D8"))
    jit_case.jit_run()


def test_AvgPool3D9():
    """test AvgPool3D9"""
    jit_case = JitTrans(case=yml.get_case_info("AvgPool3D9"))
    jit_case.jit_run()


def test_AvgPool3D10():
    """test AvgPool3D10"""
    jit_case = JitTrans(case=yml.get_case_info("AvgPool3D10"))
    jit_case.jit_run()


def test_AvgPool3D11():
    """test AvgPool3D11"""
    jit_case = JitTrans(case=yml.get_case_info("AvgPool3D11"))
    jit_case.jit_run()


def test_AvgPool3D12():
    """test AvgPool3D12"""
    jit_case = JitTrans(case=yml.get_case_info("AvgPool3D12"))
    jit_case.jit_run()
