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


def test_adaptive_max_pool1d_base():
    """test adaptive_max_pool1d_base"""
    jit_case = JitTrans(case=yml.get_case_info("adaptive_max_pool1d_base"))
    jit_case.jit_run()


def test_adaptive_max_pool1d_0():
    """test adaptive_max_pool1d_0"""
    jit_case = JitTrans(case=yml.get_case_info("adaptive_max_pool1d_0"))
    jit_case.jit_run()


def test_adaptive_max_pool1d_1():
    """test adaptive_max_pool1d_1"""
    jit_case = JitTrans(case=yml.get_case_info("adaptive_max_pool1d_1"))
    jit_case.jit_run()


def test_adaptive_max_pool1d_2():
    """test adaptive_max_pool1d_2"""
    jit_case = JitTrans(case=yml.get_case_info("adaptive_max_pool1d_2"))
    jit_case.jit_run()


def test_max_pool1d_base():
    """test max_pool1d_base"""
    jit_case = JitTrans(case=yml.get_case_info("max_pool1d_base"))
    jit_case.jit_run()


def test_max_pool1d_0():
    """test max_pool1d_0"""
    jit_case = JitTrans(case=yml.get_case_info("max_pool1d_0"))
    jit_case.jit_run()


def test_max_pool1d_1():
    """test max_pool1d_1"""
    jit_case = JitTrans(case=yml.get_case_info("max_pool1d_1"))
    jit_case.jit_run()


def test_max_pool1d_2():
    """test max_pool1d_2"""
    jit_case = JitTrans(case=yml.get_case_info("max_pool1d_2"))
    jit_case.jit_run()


def test_max_pool1d_3():
    """test max_pool1d_3"""
    jit_case = JitTrans(case=yml.get_case_info("max_pool1d_3"))
    jit_case.jit_run()


def test_max_pool1d_4():
    """test max_pool1d_4"""
    jit_case = JitTrans(case=yml.get_case_info("max_pool1d_4"))
    jit_case.jit_run()


def test_max_pool1d_5():
    """test max_pool1d_5"""
    jit_case = JitTrans(case=yml.get_case_info("max_pool1d_5"))
    jit_case.jit_run()


def test_max_pool1d_6():
    """test max_pool1d_6"""
    jit_case = JitTrans(case=yml.get_case_info("max_pool1d_6"))
    jit_case.jit_run()


def test_max_pool1d_7():
    """test max_pool1d_7"""
    jit_case = JitTrans(case=yml.get_case_info("max_pool1d_7"))
    jit_case.jit_run()


def test_max_pool1d_8():
    """test max_pool1d_8"""
    jit_case = JitTrans(case=yml.get_case_info("max_pool1d_8"))
    jit_case.jit_run()


def test_max_pool1d_9():
    """test max_pool1d_9"""
    jit_case = JitTrans(case=yml.get_case_info("max_pool1d_9"))
    jit_case.jit_run()


def test_max_pool1d_10():
    """test max_pool1d_10"""
    jit_case = JitTrans(case=yml.get_case_info("max_pool1d_10"))
    jit_case.jit_run()


def test_max_pool1d_11():
    """test max_pool1d_11"""
    jit_case = JitTrans(case=yml.get_case_info("max_pool1d_11"))
    jit_case.jit_run()
