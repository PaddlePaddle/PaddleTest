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


def test_Conv2D_base():
    """test Conv2D_base"""
    jit_case = JitTrans(case=yml.get_case_info("Conv2D_base"))
    jit_case.jit_run()


def test_Conv2D0():
    """test Conv2D0"""
    jit_case = JitTrans(case=yml.get_case_info("Conv2D0"))
    jit_case.jit_run()


def test_Conv2D1():
    """test Conv2D1"""
    jit_case = JitTrans(case=yml.get_case_info("Conv2D1"))
    jit_case.jit_run()


def test_Conv2D2():
    """test Conv2D2"""
    jit_case = JitTrans(case=yml.get_case_info("Conv2D2"))
    jit_case.jit_run()


def test_Conv2D3():
    """test Conv2D3"""
    jit_case = JitTrans(case=yml.get_case_info("Conv2D3"))
    jit_case.jit_run()


def test_Conv2D4():
    """test Conv2D4"""
    jit_case = JitTrans(case=yml.get_case_info("Conv2D4"))
    jit_case.jit_run()


def test_Conv2D5():
    """test Conv2D5"""
    jit_case = JitTrans(case=yml.get_case_info("Conv2D5"))
    jit_case.jit_run()


def test_Conv2D6():
    """test Conv2D6"""
    jit_case = JitTrans(case=yml.get_case_info("Conv2D6"))
    jit_case.jit_run()


def test_Conv2D7():
    """test Conv2D7"""
    jit_case = JitTrans(case=yml.get_case_info("Conv2D7"))
    jit_case.jit_run()


def test_Conv2D8():
    """test Conv2D8"""
    jit_case = JitTrans(case=yml.get_case_info("Conv2D8"))
    jit_case.jit_run()


def test_Conv2D9():
    """test Conv2D9"""
    jit_case = JitTrans(case=yml.get_case_info("Conv2D9"))
    jit_case.jit_run()


def test_Conv2D10():
    """test Conv2D10"""
    jit_case = JitTrans(case=yml.get_case_info("Conv2D10"))
    jit_case.jit_run()


def test_Conv2D11():
    """test Conv2D11"""
    jit_case = JitTrans(case=yml.get_case_info("Conv2D11"))
    jit_case.jit_run()


def test_Conv2D12():
    """test Conv2D12"""
    jit_case = JitTrans(case=yml.get_case_info("Conv2D12"))
    jit_case.jit_run()


def test_Conv2D13():
    """test Conv2D13"""
    jit_case = JitTrans(case=yml.get_case_info("Conv2D13"))
    jit_case.jit_run()


def test_Conv2D14():
    """test Conv2D14"""
    jit_case = JitTrans(case=yml.get_case_info("Conv2D14"))
    jit_case.jit_run()


def test_Conv2D15():
    """test Conv2D15"""
    jit_case = JitTrans(case=yml.get_case_info("Conv2D15"))
    jit_case.jit_run()


def test_Conv2DTranspose_base():
    """test Conv2DTranspose_base"""
    jit_case = JitTrans(case=yml.get_case_info("Conv2DTranspose_base"))
    jit_case.jit_run()


def test_Conv2DTranspose():
    """test Conv2DTranspose"""
    jit_case = JitTrans(case=yml.get_case_info("Conv2DTranspose"))
    jit_case.jit_run()


def test_Conv2DTranspose0():
    """test Conv2DTranspose0"""
    jit_case = JitTrans(case=yml.get_case_info("Conv2DTranspose0"))
    jit_case.jit_run()


def test_Conv2DTranspose1():
    """test Conv2DTranspose1"""
    jit_case = JitTrans(case=yml.get_case_info("Conv2DTranspose1"))
    jit_case.jit_run()


def test_Conv2DTranspose2():
    """test Conv2DTranspose2"""
    jit_case = JitTrans(case=yml.get_case_info("Conv2DTranspose2"))
    jit_case.jit_run()


def test_Conv2DTranspose3():
    """test Conv2DTranspose3"""
    jit_case = JitTrans(case=yml.get_case_info("Conv2DTranspose3"))
    jit_case.jit_run()


def test_Conv2DTranspose4():
    """test Conv2DTranspose4"""
    jit_case = JitTrans(case=yml.get_case_info("Conv2DTranspose4"))
    jit_case.jit_run()


def test_Conv2DTranspose5():
    """test Conv2DTranspose5"""
    jit_case = JitTrans(case=yml.get_case_info("Conv2DTranspose5"))
    jit_case.jit_run()


def test_Conv2DTranspose6():
    """test Conv2DTranspose6"""
    jit_case = JitTrans(case=yml.get_case_info("Conv2DTranspose6"))
    jit_case.jit_run()
