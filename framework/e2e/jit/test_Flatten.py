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


def test_Flatten_base():
    """test Flatten_base"""
    jit_case = JitTrans(case=yml.get_case_info("Flatten_base"))
    jit_case.jit_run()


def test_Flatten_0():
    """test Flatten_0"""
    jit_case = JitTrans(case=yml.get_case_info("Flatten_0"))
    jit_case.jit_run()


def test_Flatten_1():
    """test Flatten_1"""
    jit_case = JitTrans(case=yml.get_case_info("Flatten_1"))
    jit_case.jit_run()


def test_Flatten_2():
    """test Flatten_2"""
    jit_case = JitTrans(case=yml.get_case_info("Flatten_2"))
    jit_case.jit_run()


def test_Flatten_3():
    """test Flatten_3"""
    jit_case = JitTrans(case=yml.get_case_info("Flatten_3"))
    jit_case.jit_run()


def test_Flatten_4():
    """test Flatten_4"""
    jit_case = JitTrans(case=yml.get_case_info("Flatten_4"))
    jit_case.jit_run()


def test_Flatten_5():
    """test Flatten_5"""
    jit_case = JitTrans(case=yml.get_case_info("Flatten_5"))
    jit_case.jit_run()


def test_Flatten_6():
    """test Flatten_6"""
    jit_case = JitTrans(case=yml.get_case_info("Flatten_6"))
    jit_case.jit_run()


def test_Flatten_7():
    """test Flatten_7"""
    jit_case = JitTrans(case=yml.get_case_info("Flatten_7"))
    jit_case.jit_run()


def test_Flatten_8():
    """test Flatten_8"""
    jit_case = JitTrans(case=yml.get_case_info("Flatten_8"))
    jit_case.jit_run()


def test_Flatten_9():
    """test Flatten_9"""
    jit_case = JitTrans(case=yml.get_case_info("Flatten_9"))
    jit_case.jit_run()


def test_Flatten_10():
    """test Flatten_10"""
    jit_case = JitTrans(case=yml.get_case_info("Flatten_10"))
    jit_case.jit_run()


def test_Flatten_11():
    """test Flatten_11"""
    jit_case = JitTrans(case=yml.get_case_info("Flatten_11"))
    jit_case.jit_run()


def test_Flatten_12():
    """test Flatten_12"""
    jit_case = JitTrans(case=yml.get_case_info("Flatten_12"))
    jit_case.jit_run()
