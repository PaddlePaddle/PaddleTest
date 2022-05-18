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
