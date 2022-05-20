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


def test_Pad1D():
    """test Pad1D"""
    jit_case = JitTrans(case=yml.get_case_info("Pad1D"))
    jit_case.jit_run()


def test_Pad1D2():
    """test Pad1D2"""
    jit_case = JitTrans(case=yml.get_case_info("Pad1D2"))
    jit_case.jit_run()


def test_Pad1D3():
    """test Pad1D3"""
    jit_case = JitTrans(case=yml.get_case_info("Pad1D3"))
    jit_case.jit_run()


def test_Pad1D4():
    """test Pad1D4"""
    jit_case = JitTrans(case=yml.get_case_info("Pad1D4"))
    jit_case.jit_run()


def test_Pad1D5():
    """test Pad1D5"""
    jit_case = JitTrans(case=yml.get_case_info("Pad1D5"))
    jit_case.jit_run()


def test_Pad1D6():
    """test Pad1D6"""
    jit_case = JitTrans(case=yml.get_case_info("Pad1D6"))
    jit_case.jit_run()


def test_Pad1D7():
    """test Pad1D7"""
    jit_case = JitTrans(case=yml.get_case_info("Pad1D7"))
    jit_case.jit_run()


def test_Pad1D8():
    """test Pad1D8"""
    jit_case = JitTrans(case=yml.get_case_info("Pad1D8"))
    jit_case.jit_run()


def test_Pad1D9():
    """test Pad1D9"""
    jit_case = JitTrans(case=yml.get_case_info("Pad1D9"))
    jit_case.jit_run()


def test_Pad1D10():
    """test Pad1D10"""
    jit_case = JitTrans(case=yml.get_case_info("Pad1D10"))
    jit_case.jit_run()


def test_Pad1D11():
    """test Pad1D11"""
    jit_case = JitTrans(case=yml.get_case_info("Pad1D11"))
    jit_case.jit_run()


def test_Pad1D12():
    """test Pad1D12"""
    jit_case = JitTrans(case=yml.get_case_info("Pad1D12"))
    jit_case.jit_run()


def test_Pad1D13():
    """test Pad1D13"""
    jit_case = JitTrans(case=yml.get_case_info("Pad1D13"))
    jit_case.jit_run()


def test_Pad1D14():
    """test Pad1D14"""
    jit_case = JitTrans(case=yml.get_case_info("Pad1D14"))
    jit_case.jit_run()


def test_Pad1D15():
    """test Pad1D15"""
    jit_case = JitTrans(case=yml.get_case_info("Pad1D15"))
    jit_case.jit_run()


def test_Pad1D16():
    """test Pad1D16"""
    jit_case = JitTrans(case=yml.get_case_info("Pad1D16"))
    jit_case.jit_run()


def test_Pad1D17():
    """test Pad1D17"""
    jit_case = JitTrans(case=yml.get_case_info("Pad1D17"))
    jit_case.jit_run()


def test_Pad1D18():
    """test Pad1D18"""
    jit_case = JitTrans(case=yml.get_case_info("Pad1D18"))
    jit_case.jit_run()


def test_Pad1D19():
    """test Pad1D19"""
    jit_case = JitTrans(case=yml.get_case_info("Pad1D19"))
    jit_case.jit_run()


def test_Pad1D20():
    """test Pad1D20"""
    jit_case = JitTrans(case=yml.get_case_info("Pad1D20"))
    jit_case.jit_run()


def test_Pad1D21():
    """test Pad1D21"""
    jit_case = JitTrans(case=yml.get_case_info("Pad1D21"))
    jit_case.jit_run()


def test_Pad1D22():
    """test Pad1D22"""
    jit_case = JitTrans(case=yml.get_case_info("Pad1D22"))
    jit_case.jit_run()


def test_Pad1D23():
    """test Pad1D23"""
    jit_case = JitTrans(case=yml.get_case_info("Pad1D23"))
    jit_case.jit_run()


def test_Pad1D24():
    """test Pad1D24"""
    jit_case = JitTrans(case=yml.get_case_info("Pad1D24"))
    jit_case.jit_run()


def test_Pad1D25():
    """test Pad1D25"""
    jit_case = JitTrans(case=yml.get_case_info("Pad1D25"))
    jit_case.jit_run()


def test_Pad1D26():
    """test Pad1D26"""
    jit_case = JitTrans(case=yml.get_case_info("Pad1D26"))
    jit_case.jit_run()


def test_Pad1D27():
    """test Pad1D27"""
    jit_case = JitTrans(case=yml.get_case_info("Pad1D27"))
    jit_case.jit_run()
