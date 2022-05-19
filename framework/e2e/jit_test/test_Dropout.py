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


def test_AlphaDropout_base():
    """test AlphaDropout_base"""
    jit_case = JitTrans(case=yml.get_case_info("AlphaDropout_base"))
    jit_case.jit_run()


def test_AlphaDropout():
    """test AlphaDropout"""
    jit_case = JitTrans(case=yml.get_case_info("AlphaDropout"))
    jit_case.jit_run()


def test_AlphaDropout1():
    """test AlphaDropout1"""
    jit_case = JitTrans(case=yml.get_case_info("AlphaDropout1"))
    jit_case.jit_run()


def test_AlphaDropout2():
    """test AlphaDropout2"""
    jit_case = JitTrans(case=yml.get_case_info("AlphaDropout2"))
    jit_case.jit_run()


def test_AlphaDropout3():
    """test AlphaDropout3"""
    jit_case = JitTrans(case=yml.get_case_info("AlphaDropout3"))
    jit_case.jit_run()


def test_AlphaDropout4():
    """test AlphaDropout4"""
    jit_case = JitTrans(case=yml.get_case_info("AlphaDropout4"))
    jit_case.jit_run()


def test_AlphaDropout5():
    """test AlphaDropout5"""
    jit_case = JitTrans(case=yml.get_case_info("AlphaDropout5"))
    jit_case.jit_run()


def test_Dropout_base():
    """test Dropout_base"""
    jit_case = JitTrans(case=yml.get_case_info("Dropout_base"))
    jit_case.jit_run()


def test_Dropout():
    """test Dropout"""
    jit_case = JitTrans(case=yml.get_case_info("Dropout"))
    jit_case.jit_run()


def test_Dropout0():
    """test Dropout0"""
    jit_case = JitTrans(case=yml.get_case_info("Dropout0"))
    jit_case.jit_run()


def test_Dropout1():
    """test Dropout1"""
    jit_case = JitTrans(case=yml.get_case_info("Dropout1"))
    jit_case.jit_run()


def test_Dropout2():
    """test Dropout2"""
    jit_case = JitTrans(case=yml.get_case_info("Dropout2"))
    jit_case.jit_run()


def test_Dropout3():
    """test Dropout3"""
    jit_case = JitTrans(case=yml.get_case_info("Dropout3"))
    jit_case.jit_run()


def test_Dropout2D_base():
    """test Dropout2D_base"""
    jit_case = JitTrans(case=yml.get_case_info("Dropout2D_base"))
    jit_case.jit_run()


def test_Dropout2D():
    """test Dropout2D"""
    jit_case = JitTrans(case=yml.get_case_info("Dropout2D"))
    jit_case.jit_run()


def test_Dropout2D0():
    """test Dropout2D0"""
    jit_case = JitTrans(case=yml.get_case_info("Dropout2D0"))
    jit_case.jit_run()


def test_Dropout2D1():
    """test Dropout2D1"""
    jit_case = JitTrans(case=yml.get_case_info("Dropout2D1"))
    jit_case.jit_run()


def test_Dropout3D_base():
    """test Dropout3D_base"""
    jit_case = JitTrans(case=yml.get_case_info("Dropout3D_base"))
    jit_case.jit_run()


def test_Dropout3D():
    """test Dropout3D"""
    jit_case = JitTrans(case=yml.get_case_info("Dropout3D"))
    jit_case.jit_run()


def test_Dropout3D0():
    """test Dropout3D0"""
    jit_case = JitTrans(case=yml.get_case_info("Dropout3D0"))
    jit_case.jit_run()


def test_Dropout3D1():
    """test Dropout3D1"""
    jit_case = JitTrans(case=yml.get_case_info("Dropout3D1"))
    jit_case.jit_run()


def test_Dropout3D2():
    """test Dropout3D2"""
    jit_case = JitTrans(case=yml.get_case_info("Dropout3D2"))
    jit_case.jit_run()


def test_Dropout3D3():
    """test Dropout3D3"""
    jit_case = JitTrans(case=yml.get_case_info("Dropout3D3"))
    jit_case.jit_run()
