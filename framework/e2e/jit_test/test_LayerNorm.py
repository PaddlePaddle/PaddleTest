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


def test_LayerNorm_base():
    """test LayerNorm_base"""
    jit_case = JitTrans(case=yml.get_case_info("LayerNorm_base"))
    jit_case.jit_run()


def test_LayerNorm():
    """test LayerNorm"""
    jit_case = JitTrans(case=yml.get_case_info("LayerNorm"))
    jit_case.jit_run()


def test_LayerNorm3():
    """test LayerNorm3"""
    jit_case = JitTrans(case=yml.get_case_info("LayerNorm3"))
    jit_case.jit_run()


def test_LayerNorm4():
    """test LayerNorm4"""
    jit_case = JitTrans(case=yml.get_case_info("LayerNorm4"))
    jit_case.jit_run()


def test_LayerNorm5():
    """test LayerNorm5"""
    jit_case = JitTrans(case=yml.get_case_info("LayerNorm5"))
    jit_case.jit_run()


def test_LayerNorm6():
    """test LayerNorm6"""
    jit_case = JitTrans(case=yml.get_case_info("LayerNorm6"))
    jit_case.jit_run()


def test_LayerNorm7():
    """test LayerNorm7"""
    jit_case = JitTrans(case=yml.get_case_info("LayerNorm7"))
    jit_case.jit_run()


def test_LayerNorm8():
    """test LayerNorm8"""
    jit_case = JitTrans(case=yml.get_case_info("LayerNorm8"))
    jit_case.jit_run()


def test_LayerNorm9():
    """test LayerNorm9"""
    jit_case = JitTrans(case=yml.get_case_info("LayerNorm9"))
    jit_case.jit_run()
