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
