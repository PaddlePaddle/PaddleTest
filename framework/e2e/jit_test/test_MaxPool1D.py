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


def test_AdaptiveMaxPool1D_base():
    """test AdaptiveMaxPool1D_base"""
    jit_case = JitTrans(case=yml.get_case_info("AdaptiveMaxPool1D_base"))
    jit_case.jit_run()


def test_AdaptiveMaxPool1D():
    """test AdaptiveMaxPool1D"""
    jit_case = JitTrans(case=yml.get_case_info("AdaptiveMaxPool1D"))
    jit_case.jit_run()


def test_AdaptiveMaxPool1D0():
    """test AdaptiveMaxPool1D0"""
    jit_case = JitTrans(case=yml.get_case_info("AdaptiveMaxPool1D0"))
    jit_case.jit_run()


def test_AdaptiveMaxPool1D2():
    """test AdaptiveMaxPool1D2"""
    jit_case = JitTrans(case=yml.get_case_info("AdaptiveMaxPool1D2"))
    jit_case.jit_run()


def test_AdaptiveMaxPool1D3():
    """test AdaptiveMaxPool1D3"""
    jit_case = JitTrans(case=yml.get_case_info("AdaptiveMaxPool1D3"))
    jit_case.jit_run()


def test_AdaptiveMaxPool1D4():
    """test AdaptiveMaxPool1D4"""
    jit_case = JitTrans(case=yml.get_case_info("AdaptiveMaxPool1D4"))
    jit_case.jit_run()


def test_MaxPool1D_base():
    """test MaxPool1D_base"""
    jit_case = JitTrans(case=yml.get_case_info("MaxPool1D_base"))
    jit_case.jit_run()


def test_MaxPool1D():
    """test MaxPool1D"""
    jit_case = JitTrans(case=yml.get_case_info("MaxPool1D"))
    jit_case.jit_run()


def test_MaxPool1D1():
    """test MaxPool1D1"""
    jit_case = JitTrans(case=yml.get_case_info("MaxPool1D1"))
    jit_case.jit_run()


def test_MaxPool1D2():
    """test MaxPool1D2"""
    jit_case = JitTrans(case=yml.get_case_info("MaxPool1D2"))
    jit_case.jit_run()


def test_MaxPool1D3():
    """test MaxPool1D3"""
    jit_case = JitTrans(case=yml.get_case_info("MaxPool1D3"))
    jit_case.jit_run()


def test_MaxPool1D4():
    """test MaxPool1D4"""
    jit_case = JitTrans(case=yml.get_case_info("MaxPool1D4"))
    jit_case.jit_run()


def test_MaxPool1D5():
    """test MaxPool1D5"""
    jit_case = JitTrans(case=yml.get_case_info("MaxPool1D5"))
    jit_case.jit_run()


def test_MaxPool1D6():
    """test MaxPool1D6"""
    jit_case = JitTrans(case=yml.get_case_info("MaxPool1D6"))
    jit_case.jit_run()


def test_MaxPool1D7():
    """test MaxPool1D7"""
    jit_case = JitTrans(case=yml.get_case_info("MaxPool1D7"))
    jit_case.jit_run()


def test_MaxPool1D8():
    """test MaxPool1D8"""
    jit_case = JitTrans(case=yml.get_case_info("MaxPool1D8"))
    jit_case.jit_run()


def test_MaxPool1D9():
    """test MaxPool1D9"""
    jit_case = JitTrans(case=yml.get_case_info("MaxPool1D9"))
    jit_case.jit_run()


def test_MaxPool1D10():
    """test MaxPool1D10"""
    jit_case = JitTrans(case=yml.get_case_info("MaxPool1D10"))
    jit_case.jit_run()
