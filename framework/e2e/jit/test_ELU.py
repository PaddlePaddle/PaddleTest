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


def test_CELU_base():
    """test CELU_base"""
    jit_case = JitTrans(case=yml.get_case_info("CELU_base"))
    jit_case.jit_run()


def test_CELU_0():
    """test CELU_0"""
    jit_case = JitTrans(case=yml.get_case_info("CELU_0"))
    jit_case.jit_run()


def test_CELU_1():
    """test CELU_1"""
    jit_case = JitTrans(case=yml.get_case_info("CELU_1"))
    jit_case.jit_run()


def test_CELU_2():
    """test CELU_2"""
    jit_case = JitTrans(case=yml.get_case_info("CELU_2"))
    jit_case.jit_run()


def test_CELU_3():
    """test CELU_3"""
    jit_case = JitTrans(case=yml.get_case_info("CELU_3"))
    jit_case.jit_run()


def test_ELU_base():
    """test ELU_base"""
    jit_case = JitTrans(case=yml.get_case_info("ELU_base"))
    jit_case.jit_run()


def test_ELU_0():
    """test ELU_0"""
    jit_case = JitTrans(case=yml.get_case_info("ELU_0"))
    jit_case.jit_run()


def test_ELU_1():
    """test ELU_1"""
    jit_case = JitTrans(case=yml.get_case_info("ELU_1"))
    jit_case.jit_run()


def test_ELU_2():
    """test ELU_2"""
    jit_case = JitTrans(case=yml.get_case_info("ELU_2"))
    jit_case.jit_run()


def test_ELU_3():
    """test ELU_3"""
    jit_case = JitTrans(case=yml.get_case_info("ELU_3"))
    jit_case.jit_run()


def test_GELU_base():
    """test GELU_base"""
    jit_case = JitTrans(case=yml.get_case_info("GELU_base"))
    jit_case.jit_run()


def test_GELU_0():
    """test GELU_0"""
    jit_case = JitTrans(case=yml.get_case_info("GELU_0"))
    jit_case.jit_run()


def test_GELU_1():
    """test GELU_1"""
    jit_case = JitTrans(case=yml.get_case_info("GELU_1"))
    jit_case.jit_run()


def test_SELU_base():
    """test SELU_base"""
    jit_case = JitTrans(case=yml.get_case_info("SELU_base"))
    jit_case.jit_run()


def test_SELU_0():
    """test SELU_0"""
    jit_case = JitTrans(case=yml.get_case_info("SELU_0"))
    jit_case.jit_run()


def test_SELU_1():
    """test SELU_1"""
    jit_case = JitTrans(case=yml.get_case_info("SELU_1"))
    jit_case.jit_run()


def test_SELU_2():
    """test SELU_2"""
    jit_case = JitTrans(case=yml.get_case_info("SELU_2"))
    jit_case.jit_run()
