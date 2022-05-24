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


def test_Pad1D_base():
    """test Pad1D_base"""
    jit_case = JitTrans(case=yml.get_case_info("Pad1D_base"))
    jit_case.jit_run()


def test_Pad1D_1():
    """test Pad1D_1"""
    jit_case = JitTrans(case=yml.get_case_info("Pad1D_1"))
    jit_case.jit_run()


def test_Pad1D_2():
    """test Pad1D_2"""
    jit_case = JitTrans(case=yml.get_case_info("Pad1D_2"))
    jit_case.jit_run()


def test_Pad1D_3():
    """test Pad1D_3"""
    jit_case = JitTrans(case=yml.get_case_info("Pad1D_3"))
    jit_case.jit_run()


def test_Pad1D_4():
    """test Pad1D_4"""
    jit_case = JitTrans(case=yml.get_case_info("Pad1D_4"))
    jit_case.jit_run()


def test_Pad1D_5():
    """test Pad1D_5"""
    jit_case = JitTrans(case=yml.get_case_info("Pad1D_5"))
    jit_case.jit_run()


def test_Pad1D_6():
    """test Pad1D_6"""
    jit_case = JitTrans(case=yml.get_case_info("Pad1D_6"))
    jit_case.jit_run()


def test_Pad1D_7():
    """test Pad1D_7"""
    jit_case = JitTrans(case=yml.get_case_info("Pad1D_7"))
    jit_case.jit_run()


def test_Pad1D_8():
    """test Pad1D_8"""
    jit_case = JitTrans(case=yml.get_case_info("Pad1D_8"))
    jit_case.jit_run()


def test_Pad1D_9():
    """test Pad1D_9"""
    jit_case = JitTrans(case=yml.get_case_info("Pad1D_9"))
    jit_case.jit_run()


def test_Pad1D_10():
    """test Pad1D_10"""
    jit_case = JitTrans(case=yml.get_case_info("Pad1D_10"))
    jit_case.jit_run()


def test_Pad1D_11():
    """test Pad1D_11"""
    jit_case = JitTrans(case=yml.get_case_info("Pad1D_11"))
    jit_case.jit_run()


def test_Pad1D_12():
    """test Pad1D_12"""
    jit_case = JitTrans(case=yml.get_case_info("Pad1D_12"))
    jit_case.jit_run()


def test_Pad1D_13():
    """test Pad1D_13"""
    jit_case = JitTrans(case=yml.get_case_info("Pad1D_13"))
    jit_case.jit_run()


def test_Pad1D_14():
    """test Pad1D_14"""
    jit_case = JitTrans(case=yml.get_case_info("Pad1D_14"))
    jit_case.jit_run()


def test_Pad1D_15():
    """test Pad1D_15"""
    jit_case = JitTrans(case=yml.get_case_info("Pad1D_15"))
    jit_case.jit_run()


def test_Pad1D_16():
    """test Pad1D_16"""
    jit_case = JitTrans(case=yml.get_case_info("Pad1D_16"))
    jit_case.jit_run()


def test_Pad1D_17():
    """test Pad1D_17"""
    jit_case = JitTrans(case=yml.get_case_info("Pad1D_17"))
    jit_case.jit_run()


def test_Pad1D_18():
    """test Pad1D_18"""
    jit_case = JitTrans(case=yml.get_case_info("Pad1D_18"))
    jit_case.jit_run()


def test_Pad1D_19():
    """test Pad1D_19"""
    jit_case = JitTrans(case=yml.get_case_info("Pad1D_19"))
    jit_case.jit_run()


def test_Pad1D_20():
    """test Pad1D_20"""
    jit_case = JitTrans(case=yml.get_case_info("Pad1D_20"))
    jit_case.jit_run()


def test_Pad1D_21():
    """test Pad1D_21"""
    jit_case = JitTrans(case=yml.get_case_info("Pad1D_21"))
    jit_case.jit_run()


def test_Pad1D_22():
    """test Pad1D_22"""
    jit_case = JitTrans(case=yml.get_case_info("Pad1D_22"))
    jit_case.jit_run()


def test_Pad1D_23():
    """test Pad1D_23"""
    jit_case = JitTrans(case=yml.get_case_info("Pad1D_23"))
    jit_case.jit_run()


def test_Pad1D_24():
    """test Pad1D_24"""
    jit_case = JitTrans(case=yml.get_case_info("Pad1D_24"))
    jit_case.jit_run()


def test_Pad1D_25():
    """test Pad1D_25"""
    jit_case = JitTrans(case=yml.get_case_info("Pad1D_25"))
    jit_case.jit_run()


def test_Pad1D_26():
    """test Pad1D_26"""
    jit_case = JitTrans(case=yml.get_case_info("Pad1D_26"))
    jit_case.jit_run()


def test_Pad1D_27():
    """test Pad1D_27"""
    jit_case = JitTrans(case=yml.get_case_info("Pad1D_27"))
    jit_case.jit_run()
