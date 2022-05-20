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


def test_AdaptiveMaxPool3D_base():
    """test AdaptiveMaxPool3D_base"""
    jit_case = JitTrans(case=yml.get_case_info("AdaptiveMaxPool3D_base"))
    jit_case.jit_run()


def test_AdaptiveMaxPool3D():
    """test AdaptiveMaxPool3D"""
    jit_case = JitTrans(case=yml.get_case_info("AdaptiveMaxPool3D"))
    jit_case.jit_run()


def test_AdaptiveMaxPool3D2():
    """test AdaptiveMaxPool3D2"""
    jit_case = JitTrans(case=yml.get_case_info("AdaptiveMaxPool3D2"))
    jit_case.jit_run()


def test_MaxPool3D_base():
    """test MaxPool3D_base"""
    jit_case = JitTrans(case=yml.get_case_info("MaxPool3D_base"))
    jit_case.jit_run()


def test_MaxPool3D():
    """test MaxPool3D"""
    jit_case = JitTrans(case=yml.get_case_info("MaxPool3D"))
    jit_case.jit_run()


def test_MaxPool3D0():
    """test MaxPool3D0"""
    jit_case = JitTrans(case=yml.get_case_info("MaxPool3D0"))
    jit_case.jit_run()


def test_MaxPool3D1():
    """test MaxPool3D1"""
    jit_case = JitTrans(case=yml.get_case_info("MaxPool3D1"))
    jit_case.jit_run()


def test_MaxPool3D2():
    """test MaxPool3D2"""
    jit_case = JitTrans(case=yml.get_case_info("MaxPool3D2"))
    jit_case.jit_run()


def test_MaxPool3D3():
    """test MaxPool3D3"""
    jit_case = JitTrans(case=yml.get_case_info("MaxPool3D3"))
    jit_case.jit_run()


def test_MaxPool3D4():
    """test MaxPool3D4"""
    jit_case = JitTrans(case=yml.get_case_info("MaxPool3D4"))
    jit_case.jit_run()


def test_MaxPool3D5():
    """test MaxPool3D5"""
    jit_case = JitTrans(case=yml.get_case_info("MaxPool3D5"))
    jit_case.jit_run()


def test_MaxPool3D6():
    """test MaxPool3D6"""
    jit_case = JitTrans(case=yml.get_case_info("MaxPool3D6"))
    jit_case.jit_run()


def test_MaxPool3D7():
    """test MaxPool3D7"""
    jit_case = JitTrans(case=yml.get_case_info("MaxPool3D7"))
    jit_case.jit_run()


def test_MaxPool3D8():
    """test MaxPool3D8"""
    jit_case = JitTrans(case=yml.get_case_info("MaxPool3D8"))
    jit_case.jit_run()


def test_MaxPool3D9():
    """test MaxPool3D9"""
    jit_case = JitTrans(case=yml.get_case_info("MaxPool3D9"))
    jit_case.jit_run()


def test_MaxPool3D10():
    """test MaxPool3D10"""
    jit_case = JitTrans(case=yml.get_case_info("MaxPool3D10"))
    jit_case.jit_run()


def test_MaxPool3D11():
    """test MaxPool3D11"""
    jit_case = JitTrans(case=yml.get_case_info("MaxPool3D11"))
    jit_case.jit_run()
