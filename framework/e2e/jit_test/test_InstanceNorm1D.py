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


def test_InstanceNorm1D_base():
    """test InstanceNorm1D_base"""
    jit_case = JitTrans(case=yml.get_case_info("InstanceNorm1D_base"))
    jit_case.jit_run()


def test_InstanceNorm1D1():
    """test InstanceNorm1D1"""
    jit_case = JitTrans(case=yml.get_case_info("InstanceNorm1D1"))
    jit_case.jit_run()


def test_InstanceNorm1D2():
    """test InstanceNorm1D2"""
    jit_case = JitTrans(case=yml.get_case_info("InstanceNorm1D2"))
    jit_case.jit_run()


def test_InstanceNorm1D3():
    """test InstanceNorm1D3"""
    jit_case = JitTrans(case=yml.get_case_info("InstanceNorm1D3"))
    jit_case.jit_run()


def test_InstanceNorm1D5():
    """test InstanceNorm1D5"""
    jit_case = JitTrans(case=yml.get_case_info("InstanceNorm1D5"))
    jit_case.jit_run()


def test_InstanceNorm1D6():
    """test InstanceNorm1D6"""
    jit_case = JitTrans(case=yml.get_case_info("InstanceNorm1D6"))
    jit_case.jit_run()


def test_InstanceNorm1D7():
    """test InstanceNorm1D7"""
    jit_case = JitTrans(case=yml.get_case_info("InstanceNorm1D7"))
    jit_case.jit_run()


def test_InstanceNorm1D8():
    """test InstanceNorm1D8"""
    jit_case = JitTrans(case=yml.get_case_info("InstanceNorm1D8"))
    jit_case.jit_run()
