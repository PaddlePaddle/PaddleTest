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


def test_AdaptiveMaxPool2D_base():
    """test AdaptiveMaxPool2D_base"""
    jit_case = JitTrans(case=yml.get_case_info("AdaptiveMaxPool2D_base"))
    jit_case.jit_run()


def test_AdaptiveMaxPool2D():
    """test AdaptiveMaxPool2D"""
    jit_case = JitTrans(case=yml.get_case_info("AdaptiveMaxPool2D"))
    jit_case.jit_run()


def test_AdaptiveMaxPool2D1():
    """test AdaptiveMaxPool2D1"""
    jit_case = JitTrans(case=yml.get_case_info("AdaptiveMaxPool2D1"))
    jit_case.jit_run()


def test_AdaptiveMaxPool2D2():
    """test AdaptiveMaxPool2D2"""
    jit_case = JitTrans(case=yml.get_case_info("AdaptiveMaxPool2D2"))
    jit_case.jit_run()


def test_MaxPool2D_base():
    """test MaxPool2D_base"""
    jit_case = JitTrans(case=yml.get_case_info("MaxPool2D_base"))
    jit_case.jit_run()


def test_MaxPool2D():
    """test MaxPool2D"""
    jit_case = JitTrans(case=yml.get_case_info("MaxPool2D"))
    jit_case.jit_run()


def test_MaxPool2D0():
    """test MaxPool2D0"""
    jit_case = JitTrans(case=yml.get_case_info("MaxPool2D0"))
    jit_case.jit_run()


def test_MaxPool2D1():
    """test MaxPool2D1"""
    jit_case = JitTrans(case=yml.get_case_info("MaxPool2D1"))
    jit_case.jit_run()


def test_MaxPool2D2():
    """test MaxPool2D2"""
    jit_case = JitTrans(case=yml.get_case_info("MaxPool2D2"))
    jit_case.jit_run()


def test_MaxPool2D3():
    """test MaxPool2D3"""
    jit_case = JitTrans(case=yml.get_case_info("MaxPool2D3"))
    jit_case.jit_run()


def test_MaxPool2D4():
    """test MaxPool2D4"""
    jit_case = JitTrans(case=yml.get_case_info("MaxPool2D4"))
    jit_case.jit_run()


def test_MaxPool2D5():
    """test MaxPool2D5"""
    jit_case = JitTrans(case=yml.get_case_info("MaxPool2D5"))
    jit_case.jit_run()


def test_MaxPool2D6():
    """test MaxPool2D6"""
    jit_case = JitTrans(case=yml.get_case_info("MaxPool2D6"))
    jit_case.jit_run()
