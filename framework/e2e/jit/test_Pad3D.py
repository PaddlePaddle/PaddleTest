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


def test_Pad3D_base():
    """test Pad3D_base"""
    jit_case = JitTrans(case=yml.get_case_info("Pad3D_base"))
    jit_case.jit_run()


def test_Pad3D_0():
    """test Pad3D_0"""
    jit_case = JitTrans(case=yml.get_case_info("Pad3D_0"))
    jit_case.jit_run()


def test_Pad3D_1():
    """test Pad3D_1"""
    jit_case = JitTrans(case=yml.get_case_info("Pad3D_1"))
    jit_case.jit_run()


def test_Pad3D_2():
    """test Pad3D_2"""
    jit_case = JitTrans(case=yml.get_case_info("Pad3D_2"))
    jit_case.jit_run()


def test_Pad3D_3():
    """test Pad3D_3"""
    jit_case = JitTrans(case=yml.get_case_info("Pad3D_3"))
    jit_case.jit_run()


def test_Pad3D_4():
    """test Pad3D_4"""
    jit_case = JitTrans(case=yml.get_case_info("Pad3D_4"))
    jit_case.jit_run()


def test_Pad3D_5():
    """test Pad3D_5"""
    jit_case = JitTrans(case=yml.get_case_info("Pad3D_5"))
    jit_case.jit_run()


def test_Pad3D_6():
    """test Pad3D_6"""
    jit_case = JitTrans(case=yml.get_case_info("Pad3D_6"))
    jit_case.jit_run()


def test_Pad3D_7():
    """test Pad3D_7"""
    jit_case = JitTrans(case=yml.get_case_info("Pad3D_7"))
    jit_case.jit_run()


def test_Pad3D_8():
    """test Pad3D_8"""
    jit_case = JitTrans(case=yml.get_case_info("Pad3D_8"))
    jit_case.jit_run()


def test_Pad3D_9():
    """test Pad3D_9"""
    jit_case = JitTrans(case=yml.get_case_info("Pad3D_9"))
    jit_case.jit_run()


def test_Pad3D_10():
    """test Pad3D_10"""
    jit_case = JitTrans(case=yml.get_case_info("Pad3D_10"))
    jit_case.jit_run()


def test_Pad3D_11():
    """test Pad3D_11"""
    jit_case = JitTrans(case=yml.get_case_info("Pad3D_11"))
    jit_case.jit_run()


def test_Pad3D_12():
    """test Pad3D_12"""
    jit_case = JitTrans(case=yml.get_case_info("Pad3D_12"))
    jit_case.jit_run()


def test_Pad3D_13():
    """test Pad3D_13"""
    jit_case = JitTrans(case=yml.get_case_info("Pad3D_13"))
    jit_case.jit_run()


def test_Pad3D_14():
    """test Pad3D_14"""
    jit_case = JitTrans(case=yml.get_case_info("Pad3D_14"))
    jit_case.jit_run()


def test_Pad3D_15():
    """test Pad3D_15"""
    jit_case = JitTrans(case=yml.get_case_info("Pad3D_15"))
    jit_case.jit_run()


def test_Pad3D_16():
    """test Pad3D_16"""
    jit_case = JitTrans(case=yml.get_case_info("Pad3D_16"))
    jit_case.jit_run()


def test_Pad3D_17():
    """test Pad3D_17"""
    jit_case = JitTrans(case=yml.get_case_info("Pad3D_17"))
    jit_case.jit_run()


def test_Pad3D_18():
    """test Pad3D_18"""
    jit_case = JitTrans(case=yml.get_case_info("Pad3D_18"))
    jit_case.jit_run()


def test_Pad3D_19():
    """test Pad3D_19"""
    jit_case = JitTrans(case=yml.get_case_info("Pad3D_19"))
    jit_case.jit_run()


def test_Pad3D_20():
    """test Pad3D_20"""
    jit_case = JitTrans(case=yml.get_case_info("Pad3D_20"))
    jit_case.jit_run()


def test_Pad3D_21():
    """test Pad3D_21"""
    jit_case = JitTrans(case=yml.get_case_info("Pad3D_21"))
    jit_case.jit_run()


def test_Pad3D_22():
    """test Pad3D_22"""
    jit_case = JitTrans(case=yml.get_case_info("Pad3D_22"))
    jit_case.jit_run()


def test_Pad3D_23():
    """test Pad3D_23"""
    jit_case = JitTrans(case=yml.get_case_info("Pad3D_23"))
    jit_case.jit_run()


def test_Pad3D_24():
    """test Pad3D_24"""
    jit_case = JitTrans(case=yml.get_case_info("Pad3D_24"))
    jit_case.jit_run()


def test_Pad3D_25():
    """test Pad3D_25"""
    jit_case = JitTrans(case=yml.get_case_info("Pad3D_25"))
    jit_case.jit_run()


def test_Pad3D_26():
    """test Pad3D_26"""
    jit_case = JitTrans(case=yml.get_case_info("Pad3D_26"))
    jit_case.jit_run()


def test_Pad3D_27():
    """test Pad3D_27"""
    jit_case = JitTrans(case=yml.get_case_info("Pad3D_27"))
    jit_case.jit_run()
