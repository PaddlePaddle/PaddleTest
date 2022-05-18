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


def test_BatchNorm_base():
    """test BatchNorm_base"""
    jit_case = JitTrans(case=yml.get_case_info("BatchNorm_base"))
    jit_case.jit_run()


def test_BatchNorm():
    """test BatchNorm"""
    jit_case = JitTrans(case=yml.get_case_info("BatchNorm"))
    jit_case.jit_run()


def test_BatchNorm0():
    """test BatchNorm0"""
    jit_case = JitTrans(case=yml.get_case_info("BatchNorm0"))
    jit_case.jit_run()


def test_BatchNorm1():
    """test BatchNorm1"""
    jit_case = JitTrans(case=yml.get_case_info("BatchNorm1"))
    jit_case.jit_run()


def test_BatchNorm2():
    """test BatchNorm2"""
    jit_case = JitTrans(case=yml.get_case_info("BatchNorm2"))
    jit_case.jit_run()


def test_BatchNorm3():
    """test BatchNorm3"""
    jit_case = JitTrans(case=yml.get_case_info("BatchNorm3"))
    jit_case.jit_run()


def test_BatchNorm4():
    """test BatchNorm4"""
    jit_case = JitTrans(case=yml.get_case_info("BatchNorm4"))
    jit_case.jit_run()


def test_BatchNorm7():
    """test BatchNorm7"""
    jit_case = JitTrans(case=yml.get_case_info("BatchNorm7"))
    jit_case.jit_run()


def test_BatchNorm8():
    """test BatchNorm8"""
    jit_case = JitTrans(case=yml.get_case_info("BatchNorm8"))
    jit_case.jit_run()


def test_BatchNorm10():
    """test BatchNorm10"""
    jit_case = JitTrans(case=yml.get_case_info("BatchNorm10"))
    jit_case.jit_run()


def test_BatchNorm11():
    """test BatchNorm11"""
    jit_case = JitTrans(case=yml.get_case_info("BatchNorm11"))
    jit_case.jit_run()


def test_BatchNorm12():
    """test BatchNorm12"""
    jit_case = JitTrans(case=yml.get_case_info("BatchNorm12"))
    jit_case.jit_run()


def test_BatchNorm13():
    """test BatchNorm13"""
    jit_case = JitTrans(case=yml.get_case_info("BatchNorm13"))
    jit_case.jit_run()


def test_BatchNorm15():
    """test BatchNorm15"""
    jit_case = JitTrans(case=yml.get_case_info("BatchNorm15"))
    jit_case.jit_run()


def test_BatchNorm17():
    """test BatchNorm17"""
    jit_case = JitTrans(case=yml.get_case_info("BatchNorm17"))
    jit_case.jit_run()


def test_BatchNorm1D_base():
    """test BatchNorm1D_base"""
    jit_case = JitTrans(case=yml.get_case_info("BatchNorm1D_base"))
    jit_case.jit_run()


def test_BatchNorm1D():
    """test BatchNorm1D"""
    jit_case = JitTrans(case=yml.get_case_info("BatchNorm1D"))
    jit_case.jit_run()


def test_BatchNorm1D2():
    """test BatchNorm1D2"""
    jit_case = JitTrans(case=yml.get_case_info("BatchNorm1D2"))
    jit_case.jit_run()


def test_BatchNorm1D3():
    """test BatchNorm1D3"""
    jit_case = JitTrans(case=yml.get_case_info("BatchNorm1D3"))
    jit_case.jit_run()


def test_BatchNorm1D4():
    """test BatchNorm1D4"""
    jit_case = JitTrans(case=yml.get_case_info("BatchNorm1D4"))
    jit_case.jit_run()


def test_BatchNorm1D5():
    """test BatchNorm1D5"""
    jit_case = JitTrans(case=yml.get_case_info("BatchNorm1D5"))
    jit_case.jit_run()


def test_BatchNorm1D6():
    """test BatchNorm1D6"""
    jit_case = JitTrans(case=yml.get_case_info("BatchNorm1D6"))
    jit_case.jit_run()


def test_BatchNorm1D7():
    """test BatchNorm1D7"""
    jit_case = JitTrans(case=yml.get_case_info("BatchNorm1D7"))
    jit_case.jit_run()


def test_BatchNorm1D8():
    """test BatchNorm1D8"""
    jit_case = JitTrans(case=yml.get_case_info("BatchNorm1D8"))
    jit_case.jit_run()


def test_BatchNorm1D9():
    """test BatchNorm1D9"""
    jit_case = JitTrans(case=yml.get_case_info("BatchNorm1D9"))
    jit_case.jit_run()


def test_BatchNorm2D_base():
    """test BatchNorm2D_base"""
    jit_case = JitTrans(case=yml.get_case_info("BatchNorm2D_base"))
    jit_case.jit_run()


def test_BatchNorm2D1():
    """test BatchNorm2D1"""
    jit_case = JitTrans(case=yml.get_case_info("BatchNorm2D1"))
    jit_case.jit_run()


def test_BatchNorm2D2():
    """test BatchNorm2D2"""
    jit_case = JitTrans(case=yml.get_case_info("BatchNorm2D2"))
    jit_case.jit_run()


def test_BatchNorm2D3():
    """test BatchNorm2D3"""
    jit_case = JitTrans(case=yml.get_case_info("BatchNorm2D3"))
    jit_case.jit_run()


def test_BatchNorm2D4():
    """test BatchNorm2D4"""
    jit_case = JitTrans(case=yml.get_case_info("BatchNorm2D4"))
    jit_case.jit_run()


def test_BatchNorm2D7():
    """test BatchNorm2D7"""
    jit_case = JitTrans(case=yml.get_case_info("BatchNorm2D7"))
    jit_case.jit_run()


def test_BatchNorm2D8():
    """test BatchNorm2D8"""
    jit_case = JitTrans(case=yml.get_case_info("BatchNorm2D8"))
    jit_case.jit_run()


def test_BatchNorm3D_base():
    """test BatchNorm3D_base"""
    jit_case = JitTrans(case=yml.get_case_info("BatchNorm3D_base"))
    jit_case.jit_run()


def test_BatchNorm3D1():
    """test BatchNorm3D1"""
    jit_case = JitTrans(case=yml.get_case_info("BatchNorm3D1"))
    jit_case.jit_run()


def test_BatchNorm3D2():
    """test BatchNorm3D2"""
    jit_case = JitTrans(case=yml.get_case_info("BatchNorm3D2"))
    jit_case.jit_run()


def test_BatchNorm3D3():
    """test BatchNorm3D3"""
    jit_case = JitTrans(case=yml.get_case_info("BatchNorm3D3"))
    jit_case.jit_run()


def test_BatchNorm3D4():
    """test BatchNorm3D4"""
    jit_case = JitTrans(case=yml.get_case_info("BatchNorm3D4"))
    jit_case.jit_run()


def test_BatchNorm3D5():
    """test BatchNorm3D5"""
    jit_case = JitTrans(case=yml.get_case_info("BatchNorm3D5"))
    jit_case.jit_run()
