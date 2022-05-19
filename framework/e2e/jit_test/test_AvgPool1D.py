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


def test_AdaptiveAvgPool1D_base():
    """test AdaptiveAvgPool1D_base"""
    jit_case = JitTrans(case=yml.get_case_info("AdaptiveAvgPool1D_base"))
    jit_case.jit_run()


def test_AdaptiveAvgPool1D0():
    """test AdaptiveAvgPool1D0"""
    jit_case = JitTrans(case=yml.get_case_info("AdaptiveAvgPool1D0"))
    jit_case.jit_run()


def test_AdaptiveAvgPool1D1():
    """test AdaptiveAvgPool1D1"""
    jit_case = JitTrans(case=yml.get_case_info("AdaptiveAvgPool1D1"))
    jit_case.jit_run()


def test_AvgPool1D_base():
    """test AvgPool1D_base"""
    jit_case = JitTrans(case=yml.get_case_info("AvgPool1D_base"))
    jit_case.jit_run()


def test_AvgPool1D():
    """test AvgPool1D"""
    jit_case = JitTrans(case=yml.get_case_info("AvgPool1D"))
    jit_case.jit_run()


def test_AvgPool1D1():
    """test AvgPool1D1"""
    jit_case = JitTrans(case=yml.get_case_info("AvgPool1D1"))
    jit_case.jit_run()


def test_AvgPool1D2():
    """test AvgPool1D2"""
    jit_case = JitTrans(case=yml.get_case_info("AvgPool1D2"))
    jit_case.jit_run()


def test_AvgPool1D3():
    """test AvgPool1D3"""
    jit_case = JitTrans(case=yml.get_case_info("AvgPool1D3"))
    jit_case.jit_run()
