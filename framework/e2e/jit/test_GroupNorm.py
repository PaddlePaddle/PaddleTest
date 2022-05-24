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


def test_GroupNorm_base():
    """test GroupNorm_base"""
    jit_case = JitTrans(case=yml.get_case_info("GroupNorm_base"))
    jit_case.jit_run()


def test_GroupNorm_0():
    """test GroupNorm_0"""
    jit_case = JitTrans(case=yml.get_case_info("GroupNorm_0"))
    jit_case.jit_run()


def test_GroupNorm_2():
    """test GroupNorm_2"""
    jit_case = JitTrans(case=yml.get_case_info("GroupNorm_2"))
    jit_case.jit_run()


def test_GroupNorm_5():
    """test GroupNorm_5"""
    jit_case = JitTrans(case=yml.get_case_info("GroupNorm_5"))
    jit_case.jit_run()


def test_GroupNorm_6():
    """test GroupNorm_6"""
    jit_case = JitTrans(case=yml.get_case_info("GroupNorm_6"))
    jit_case.jit_run()


def test_GroupNorm_7():
    """test GroupNorm_7"""
    jit_case = JitTrans(case=yml.get_case_info("GroupNorm_7"))
    jit_case.jit_run()


def test_GroupNorm_8():
    """test GroupNorm_8"""
    jit_case = JitTrans(case=yml.get_case_info("GroupNorm_8"))
    jit_case.jit_run()


def test_GroupNorm_9():
    """test GroupNorm_9"""
    jit_case = JitTrans(case=yml.get_case_info("GroupNorm_9"))
    jit_case.jit_run()
