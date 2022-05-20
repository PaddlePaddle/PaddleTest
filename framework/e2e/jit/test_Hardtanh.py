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


def test_Hardtanh_base():
    """test Hardtanh_base"""
    jit_case = JitTrans(case=yml.get_case_info("Hardtanh_base"))
    jit_case.jit_run()


def test_Hardtanh():
    """test Hardtanh"""
    jit_case = JitTrans(case=yml.get_case_info("Hardtanh"))
    jit_case.jit_run()


def test_Hardtanh0():
    """test Hardtanh0"""
    jit_case = JitTrans(case=yml.get_case_info("Hardtanh0"))
    jit_case.jit_run()


def test_Hardtanh1():
    """test Hardtanh1"""
    jit_case = JitTrans(case=yml.get_case_info("Hardtanh1"))
    jit_case.jit_run()


def test_Hardtanh2():
    """test Hardtanh2"""
    jit_case = JitTrans(case=yml.get_case_info("Hardtanh2"))
    jit_case.jit_run()


def test_Hardtanh3():
    """test Hardtanh3"""
    jit_case = JitTrans(case=yml.get_case_info("Hardtanh3"))
    jit_case.jit_run()


def test_Hardtanh4():
    """test Hardtanh4"""
    jit_case = JitTrans(case=yml.get_case_info("Hardtanh4"))
    jit_case.jit_run()


def test_Hardtanh5():
    """test Hardtanh5"""
    jit_case = JitTrans(case=yml.get_case_info("Hardtanh5"))
    jit_case.jit_run()


def test_Hardtanh6():
    """test Hardtanh6"""
    jit_case = JitTrans(case=yml.get_case_info("Hardtanh6"))
    jit_case.jit_run()
