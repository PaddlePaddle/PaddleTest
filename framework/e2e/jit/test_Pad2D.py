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


def test_Pad2D_base():
    """test Pad2D_base"""
    jit_case = JitTrans(case=yml.get_case_info("Pad2D_base"))
    jit_case.jit_run()


def test_Pad2D():
    """test Pad2D"""
    jit_case = JitTrans(case=yml.get_case_info("Pad2D"))
    jit_case.jit_run()


def test_Pad2D0():
    """test Pad2D0"""
    jit_case = JitTrans(case=yml.get_case_info("Pad2D0"))
    jit_case.jit_run()


def test_Pad2D2():
    """test Pad2D2"""
    jit_case = JitTrans(case=yml.get_case_info("Pad2D2"))
    jit_case.jit_run()


def test_Pad2D3():
    """test Pad2D3"""
    jit_case = JitTrans(case=yml.get_case_info("Pad2D3"))
    jit_case.jit_run()


def test_Pad2D4():
    """test Pad2D4"""
    jit_case = JitTrans(case=yml.get_case_info("Pad2D4"))
    jit_case.jit_run()


def test_Pad2D5():
    """test Pad2D5"""
    jit_case = JitTrans(case=yml.get_case_info("Pad2D5"))
    jit_case.jit_run()


def test_Pad2D6():
    """test Pad2D6"""
    jit_case = JitTrans(case=yml.get_case_info("Pad2D6"))
    jit_case.jit_run()


def test_Pad2D7():
    """test Pad2D7"""
    jit_case = JitTrans(case=yml.get_case_info("Pad2D7"))
    jit_case.jit_run()


def test_Pad2D8():
    """test Pad2D8"""
    jit_case = JitTrans(case=yml.get_case_info("Pad2D8"))
    jit_case.jit_run()


def test_Pad2D9():
    """test Pad2D9"""
    jit_case = JitTrans(case=yml.get_case_info("Pad2D9"))
    jit_case.jit_run()


def test_Pad2D10():
    """test Pad2D10"""
    jit_case = JitTrans(case=yml.get_case_info("Pad2D10"))
    jit_case.jit_run()


def test_Pad2D11():
    """test Pad2D11"""
    jit_case = JitTrans(case=yml.get_case_info("Pad2D11"))
    jit_case.jit_run()


def test_Pad2D12():
    """test Pad2D12"""
    jit_case = JitTrans(case=yml.get_case_info("Pad2D12"))
    jit_case.jit_run()


def test_Pad2D13():
    """test Pad2D13"""
    jit_case = JitTrans(case=yml.get_case_info("Pad2D13"))
    jit_case.jit_run()


def test_Pad2D14():
    """test Pad2D14"""
    jit_case = JitTrans(case=yml.get_case_info("Pad2D14"))
    jit_case.jit_run()


def test_Pad2D15():
    """test Pad2D15"""
    jit_case = JitTrans(case=yml.get_case_info("Pad2D15"))
    jit_case.jit_run()


def test_Pad2D16():
    """test Pad2D16"""
    jit_case = JitTrans(case=yml.get_case_info("Pad2D16"))
    jit_case.jit_run()


def test_Pad2D17():
    """test Pad2D17"""
    jit_case = JitTrans(case=yml.get_case_info("Pad2D17"))
    jit_case.jit_run()


def test_Pad2D18():
    """test Pad2D18"""
    jit_case = JitTrans(case=yml.get_case_info("Pad2D18"))
    jit_case.jit_run()


def test_Pad2D19():
    """test Pad2D19"""
    jit_case = JitTrans(case=yml.get_case_info("Pad2D19"))
    jit_case.jit_run()


def test_Pad2D20():
    """test Pad2D20"""
    jit_case = JitTrans(case=yml.get_case_info("Pad2D20"))
    jit_case.jit_run()


def test_Pad2D21():
    """test Pad2D21"""
    jit_case = JitTrans(case=yml.get_case_info("Pad2D21"))
    jit_case.jit_run()


def test_Pad2D22():
    """test Pad2D22"""
    jit_case = JitTrans(case=yml.get_case_info("Pad2D22"))
    jit_case.jit_run()


def test_Pad2D23():
    """test Pad2D23"""
    jit_case = JitTrans(case=yml.get_case_info("Pad2D23"))
    jit_case.jit_run()


def test_Pad2D24():
    """test Pad2D24"""
    jit_case = JitTrans(case=yml.get_case_info("Pad2D24"))
    jit_case.jit_run()


def test_Pad2D25():
    """test Pad2D25"""
    jit_case = JitTrans(case=yml.get_case_info("Pad2D25"))
    jit_case.jit_run()


def test_Pad2D26():
    """test Pad2D26"""
    jit_case = JitTrans(case=yml.get_case_info("Pad2D26"))
    jit_case.jit_run()


def test_Pad2D27():
    """test Pad2D27"""
    jit_case = JitTrans(case=yml.get_case_info("Pad2D27"))
    jit_case.jit_run()
