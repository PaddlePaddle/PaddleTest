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


def test_GroupNorm():
    """test GroupNorm"""
    jit_case = JitTrans(case=yml.get_case_info("GroupNorm"))
    jit_case.jit_run()


def test_GroupNorm2():
    """test GroupNorm2"""
    jit_case = JitTrans(case=yml.get_case_info("GroupNorm2"))
    jit_case.jit_run()


def test_GroupNorm5():
    """test GroupNorm5"""
    jit_case = JitTrans(case=yml.get_case_info("GroupNorm5"))
    jit_case.jit_run()


def test_GroupNorm6():
    """test GroupNorm6"""
    jit_case = JitTrans(case=yml.get_case_info("GroupNorm6"))
    jit_case.jit_run()


def test_GroupNorm7():
    """test GroupNorm7"""
    jit_case = JitTrans(case=yml.get_case_info("GroupNorm7"))
    jit_case.jit_run()


def test_GroupNorm8():
    """test GroupNorm8"""
    jit_case = JitTrans(case=yml.get_case_info("GroupNorm8"))
    jit_case.jit_run()


def test_GroupNorm9():
    """test GroupNorm9"""
    jit_case = JitTrans(case=yml.get_case_info("GroupNorm9"))
    jit_case.jit_run()
