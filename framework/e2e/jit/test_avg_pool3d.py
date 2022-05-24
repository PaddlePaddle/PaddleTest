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


def test_adaptive_avg_pool3d_base():
    """test adaptive_avg_pool3d_base"""
    jit_case = JitTrans(case=yml.get_case_info("adaptive_avg_pool3d_base"))
    jit_case.jit_run()


def test_adaptive_avg_pool3d_0():
    """test adaptive_avg_pool3d_0"""
    jit_case = JitTrans(case=yml.get_case_info("adaptive_avg_pool3d_0"))
    jit_case.jit_run()


def test_adaptive_avg_pool3d_1():
    """test adaptive_avg_pool3d_1"""
    jit_case = JitTrans(case=yml.get_case_info("adaptive_avg_pool3d_1"))
    jit_case.jit_run()


def test_adaptive_avg_pool3d_2():
    """test adaptive_avg_pool3d_2"""
    jit_case = JitTrans(case=yml.get_case_info("adaptive_avg_pool3d_2"))
    jit_case.jit_run()


def test_adaptive_avg_pool3d_3():
    """test adaptive_avg_pool3d_3"""
    jit_case = JitTrans(case=yml.get_case_info("adaptive_avg_pool3d_3"))
    jit_case.jit_run()


def test_adaptive_avg_pool3d_4():
    """test adaptive_avg_pool3d_4"""
    jit_case = JitTrans(case=yml.get_case_info("adaptive_avg_pool3d_4"))
    jit_case.jit_run()


def test_adaptive_avg_pool3d_5():
    """test adaptive_avg_pool3d_5"""
    jit_case = JitTrans(case=yml.get_case_info("adaptive_avg_pool3d_5"))
    jit_case.jit_run()


def test_adaptive_avg_pool3d_6():
    """test adaptive_avg_pool3d_6"""
    jit_case = JitTrans(case=yml.get_case_info("adaptive_avg_pool3d_6"))
    jit_case.jit_run()


def test_adaptive_avg_pool3d_7():
    """test adaptive_avg_pool3d_7"""
    jit_case = JitTrans(case=yml.get_case_info("adaptive_avg_pool3d_7"))
    jit_case.jit_run()


def test_adaptive_avg_pool3d_8():
    """test adaptive_avg_pool3d_8"""
    jit_case = JitTrans(case=yml.get_case_info("adaptive_avg_pool3d_8"))
    jit_case.jit_run()


def test_avg_pool3d_base():
    """test avg_pool3d_base"""
    jit_case = JitTrans(case=yml.get_case_info("avg_pool3d_base"))
    jit_case.jit_run()


def test_avg_pool3d_0():
    """test avg_pool3d_0"""
    jit_case = JitTrans(case=yml.get_case_info("avg_pool3d_0"))
    jit_case.jit_run()


def test_avg_pool3d_1():
    """test avg_pool3d_1"""
    jit_case = JitTrans(case=yml.get_case_info("avg_pool3d_1"))
    jit_case.jit_run()


def test_avg_pool3d_2():
    """test avg_pool3d_2"""
    jit_case = JitTrans(case=yml.get_case_info("avg_pool3d_2"))
    jit_case.jit_run()


def test_avg_pool3d_3():
    """test avg_pool3d_3"""
    jit_case = JitTrans(case=yml.get_case_info("avg_pool3d_3"))
    jit_case.jit_run()


def test_avg_pool3d_4():
    """test avg_pool3d_4"""
    jit_case = JitTrans(case=yml.get_case_info("avg_pool3d_4"))
    jit_case.jit_run()


def test_avg_pool3d_5():
    """test avg_pool3d_5"""
    jit_case = JitTrans(case=yml.get_case_info("avg_pool3d_5"))
    jit_case.jit_run()


def test_avg_pool3d_6():
    """test avg_pool3d_6"""
    jit_case = JitTrans(case=yml.get_case_info("avg_pool3d_6"))
    jit_case.jit_run()


def test_avg_pool3d_7():
    """test avg_pool3d_7"""
    jit_case = JitTrans(case=yml.get_case_info("avg_pool3d_7"))
    jit_case.jit_run()


def test_avg_pool3d_8():
    """test avg_pool3d_8"""
    jit_case = JitTrans(case=yml.get_case_info("avg_pool3d_8"))
    jit_case.jit_run()


def test_avg_pool3d_9():
    """test avg_pool3d_9"""
    jit_case = JitTrans(case=yml.get_case_info("avg_pool3d_9"))
    jit_case.jit_run()


def test_avg_pool3d_10():
    """test avg_pool3d_10"""
    jit_case = JitTrans(case=yml.get_case_info("avg_pool3d_10"))
    jit_case.jit_run()


def test_avg_pool3d_11():
    """test avg_pool3d_11"""
    jit_case = JitTrans(case=yml.get_case_info("avg_pool3d_11"))
    jit_case.jit_run()


def test_avg_pool3d_12():
    """test avg_pool3d_12"""
    jit_case = JitTrans(case=yml.get_case_info("avg_pool3d_12"))
    jit_case.jit_run()


def test_avg_pool3d_13():
    """test avg_pool3d_13"""
    jit_case = JitTrans(case=yml.get_case_info("avg_pool3d_13"))
    jit_case.jit_run()
