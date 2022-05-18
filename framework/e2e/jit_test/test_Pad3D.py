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


def test_Pad3D():
    """test Pad3D"""
    jit_case = JitTrans(case=yml.get_case_info("Pad3D"))
    jit_case.jit_run()


def test_Pad3D0():
    """test Pad3D0"""
    jit_case = JitTrans(case=yml.get_case_info("Pad3D0"))
    jit_case.jit_run()


def test_Pad3D2():
    """test Pad3D2"""
    jit_case = JitTrans(case=yml.get_case_info("Pad3D2"))
    jit_case.jit_run()


def test_Pad3D3():
    """test Pad3D3"""
    jit_case = JitTrans(case=yml.get_case_info("Pad3D3"))
    jit_case.jit_run()


def test_Pad3D4():
    """test Pad3D4"""
    jit_case = JitTrans(case=yml.get_case_info("Pad3D4"))
    jit_case.jit_run()


def test_Pad3D5():
    """test Pad3D5"""
    jit_case = JitTrans(case=yml.get_case_info("Pad3D5"))
    jit_case.jit_run()


def test_Pad3D6():
    """test Pad3D6"""
    jit_case = JitTrans(case=yml.get_case_info("Pad3D6"))
    jit_case.jit_run()


def test_Pad3D7():
    """test Pad3D7"""
    jit_case = JitTrans(case=yml.get_case_info("Pad3D7"))
    jit_case.jit_run()


def test_Pad3D8():
    """test Pad3D8"""
    jit_case = JitTrans(case=yml.get_case_info("Pad3D8"))
    jit_case.jit_run()


def test_Pad3D9():
    """test Pad3D9"""
    jit_case = JitTrans(case=yml.get_case_info("Pad3D9"))
    jit_case.jit_run()


def test_Pad3D10():
    """test Pad3D10"""
    jit_case = JitTrans(case=yml.get_case_info("Pad3D10"))
    jit_case.jit_run()


def test_Pad3D11():
    """test Pad3D11"""
    jit_case = JitTrans(case=yml.get_case_info("Pad3D11"))
    jit_case.jit_run()


def test_Pad3D12():
    """test Pad3D12"""
    jit_case = JitTrans(case=yml.get_case_info("Pad3D12"))
    jit_case.jit_run()


def test_Pad3D13():
    """test Pad3D13"""
    jit_case = JitTrans(case=yml.get_case_info("Pad3D13"))
    jit_case.jit_run()


def test_Pad3D14():
    """test Pad3D14"""
    jit_case = JitTrans(case=yml.get_case_info("Pad3D14"))
    jit_case.jit_run()


def test_Pad3D15():
    """test Pad3D15"""
    jit_case = JitTrans(case=yml.get_case_info("Pad3D15"))
    jit_case.jit_run()


def test_Pad3D16():
    """test Pad3D16"""
    jit_case = JitTrans(case=yml.get_case_info("Pad3D16"))
    jit_case.jit_run()


def test_Pad3D17():
    """test Pad3D17"""
    jit_case = JitTrans(case=yml.get_case_info("Pad3D17"))
    jit_case.jit_run()


def test_Pad3D18():
    """test Pad3D18"""
    jit_case = JitTrans(case=yml.get_case_info("Pad3D18"))
    jit_case.jit_run()


def test_Pad3D19():
    """test Pad3D19"""
    jit_case = JitTrans(case=yml.get_case_info("Pad3D19"))
    jit_case.jit_run()


def test_Pad3D20():
    """test Pad3D20"""
    jit_case = JitTrans(case=yml.get_case_info("Pad3D20"))
    jit_case.jit_run()


def test_Pad3D21():
    """test Pad3D21"""
    jit_case = JitTrans(case=yml.get_case_info("Pad3D21"))
    jit_case.jit_run()


def test_Pad3D22():
    """test Pad3D22"""
    jit_case = JitTrans(case=yml.get_case_info("Pad3D22"))
    jit_case.jit_run()


def test_Pad3D23():
    """test Pad3D23"""
    jit_case = JitTrans(case=yml.get_case_info("Pad3D23"))
    jit_case.jit_run()


def test_Pad3D24():
    """test Pad3D24"""
    jit_case = JitTrans(case=yml.get_case_info("Pad3D24"))
    jit_case.jit_run()


def test_Pad3D25():
    """test Pad3D25"""
    jit_case = JitTrans(case=yml.get_case_info("Pad3D25"))
    jit_case.jit_run()


def test_Pad3D26():
    """test Pad3D26"""
    jit_case = JitTrans(case=yml.get_case_info("Pad3D26"))
    jit_case.jit_run()


def test_Pad3D27():
    """test Pad3D27"""
    jit_case = JitTrans(case=yml.get_case_info("Pad3D27"))
    jit_case.jit_run()
