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


def test_CELU():
    """test CELU"""
    jit_case = JitTrans(case=yml.get_case_info("CELU"))
    jit_case.jit_run()


def test_CELU0():
    """test CELU0"""
    jit_case = JitTrans(case=yml.get_case_info("CELU0"))
    jit_case.jit_run()


def test_CELU1():
    """test CELU1"""
    jit_case = JitTrans(case=yml.get_case_info("CELU1"))
    jit_case.jit_run()


def test_CELU2():
    """test CELU2"""
    jit_case = JitTrans(case=yml.get_case_info("CELU2"))
    jit_case.jit_run()


def test_ELU_base():
    """test ELU_base"""
    jit_case = JitTrans(case=yml.get_case_info("ELU_base"))
    jit_case.jit_run()


def test_ELU():
    """test ELU"""
    jit_case = JitTrans(case=yml.get_case_info("ELU"))
    jit_case.jit_run()


def test_ELU0():
    """test ELU0"""
    jit_case = JitTrans(case=yml.get_case_info("ELU0"))
    jit_case.jit_run()


def test_ELU1():
    """test ELU1"""
    jit_case = JitTrans(case=yml.get_case_info("ELU1"))
    jit_case.jit_run()


def test_ELU2():
    """test ELU2"""
    jit_case = JitTrans(case=yml.get_case_info("ELU2"))
    jit_case.jit_run()


def test_GELU_base():
    """test GELU_base"""
    jit_case = JitTrans(case=yml.get_case_info("GELU_base"))
    jit_case.jit_run()


def test_GELU():
    """test GELU"""
    jit_case = JitTrans(case=yml.get_case_info("GELU"))
    jit_case.jit_run()


def test_GELU1():
    """test GELU1"""
    jit_case = JitTrans(case=yml.get_case_info("GELU1"))
    jit_case.jit_run()


def test_SELU_base():
    """test SELU_base"""
    jit_case = JitTrans(case=yml.get_case_info("SELU_base"))
    jit_case.jit_run()


def test_SELU():
    """test SELU"""
    jit_case = JitTrans(case=yml.get_case_info("SELU"))
    jit_case.jit_run()


def test_SELU0():
    """test SELU0"""
    jit_case = JitTrans(case=yml.get_case_info("SELU0"))
    jit_case.jit_run()


def test_SELU1():
    """test SELU1"""
    jit_case = JitTrans(case=yml.get_case_info("SELU1"))
    jit_case.jit_run()
