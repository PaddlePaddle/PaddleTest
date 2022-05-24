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


def test_AlphaDropout_0():
    """test AlphaDropout_0"""
    jit_case = JitTrans(case=yml.get_case_info("AlphaDropout_0"))
    jit_case.jit_run()


def test_AlphaDropout_1():
    """test AlphaDropout_1"""
    jit_case = JitTrans(case=yml.get_case_info("AlphaDropout_1"))
    jit_case.jit_run()


def test_AlphaDropout_2():
    """test AlphaDropout_2"""
    jit_case = JitTrans(case=yml.get_case_info("AlphaDropout_2"))
    jit_case.jit_run()


def test_AlphaDropout_3():
    """test AlphaDropout_3"""
    jit_case = JitTrans(case=yml.get_case_info("AlphaDropout_3"))
    jit_case.jit_run()


def test_AlphaDropout_4():
    """test AlphaDropout_4"""
    jit_case = JitTrans(case=yml.get_case_info("AlphaDropout_4"))
    jit_case.jit_run()


def test_AlphaDropout_5():
    """test AlphaDropout_5"""
    jit_case = JitTrans(case=yml.get_case_info("AlphaDropout_5"))
    jit_case.jit_run()


def test_Dropout_base():
    """test Dropout_base"""
    jit_case = JitTrans(case=yml.get_case_info("Dropout_base"))
    jit_case.jit_run()


def test_Dropout_0():
    """test Dropout_0"""
    jit_case = JitTrans(case=yml.get_case_info("Dropout_0"))
    jit_case.jit_run()


def test_Dropout_1():
    """test Dropout_1"""
    jit_case = JitTrans(case=yml.get_case_info("Dropout_1"))
    jit_case.jit_run()


def test_Dropout_2():
    """test Dropout_2"""
    jit_case = JitTrans(case=yml.get_case_info("Dropout_2"))
    jit_case.jit_run()


def test_Dropout_3():
    """test Dropout_3"""
    jit_case = JitTrans(case=yml.get_case_info("Dropout_3"))
    jit_case.jit_run()


def test_Dropout_4():
    """test Dropout_4"""
    jit_case = JitTrans(case=yml.get_case_info("Dropout_4"))
    jit_case.jit_run()
