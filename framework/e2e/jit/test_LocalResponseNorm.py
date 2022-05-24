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


def test_LocalResponseNorm_base():
    """test LocalResponseNorm_base"""
    jit_case = JitTrans(case=yml.get_case_info("LocalResponseNorm_base"))
    jit_case.jit_run()


def test_LocalResponseNorm_0():
    """test LocalResponseNorm_0"""
    jit_case = JitTrans(case=yml.get_case_info("LocalResponseNorm_0"))
    jit_case.jit_run()


def test_LocalResponseNorm_1():
    """test LocalResponseNorm_1"""
    jit_case = JitTrans(case=yml.get_case_info("LocalResponseNorm_1"))
    jit_case.jit_run()


def test_LocalResponseNorm_2():
    """test LocalResponseNorm_2"""
    jit_case = JitTrans(case=yml.get_case_info("LocalResponseNorm_2"))
    jit_case.jit_run()


def test_LocalResponseNorm_3():
    """test LocalResponseNorm_3"""
    jit_case = JitTrans(case=yml.get_case_info("LocalResponseNorm_3"))
    jit_case.jit_run()


def test_LocalResponseNorm_4():
    """test LocalResponseNorm_4"""
    jit_case = JitTrans(case=yml.get_case_info("LocalResponseNorm_4"))
    jit_case.jit_run()


def test_LocalResponseNorm_5():
    """test LocalResponseNorm_5"""
    jit_case = JitTrans(case=yml.get_case_info("LocalResponseNorm_5"))
    jit_case.jit_run()
