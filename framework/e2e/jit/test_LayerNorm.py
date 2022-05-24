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


def test_LayerNorm_0():
    """test LayerNorm_0"""
    jit_case = JitTrans(case=yml.get_case_info("LayerNorm_0"))
    jit_case.jit_run()


def test_LayerNorm_3():
    """test LayerNorm_3"""
    jit_case = JitTrans(case=yml.get_case_info("LayerNorm_3"))
    jit_case.jit_run()


def test_LayerNorm_4():
    """test LayerNorm_4"""
    jit_case = JitTrans(case=yml.get_case_info("LayerNorm_4"))
    jit_case.jit_run()


def test_LayerNorm_5():
    """test LayerNorm_5"""
    jit_case = JitTrans(case=yml.get_case_info("LayerNorm_5"))
    jit_case.jit_run()


def test_LayerNorm_6():
    """test LayerNorm_6"""
    jit_case = JitTrans(case=yml.get_case_info("LayerNorm_6"))
    jit_case.jit_run()


def test_LayerNorm_7():
    """test LayerNorm_7"""
    jit_case = JitTrans(case=yml.get_case_info("LayerNorm_7"))
    jit_case.jit_run()


def test_LayerNorm_8():
    """test LayerNorm_8"""
    jit_case = JitTrans(case=yml.get_case_info("LayerNorm_8"))
    jit_case.jit_run()


def test_LayerNorm_9():
    """test LayerNorm_9"""
    jit_case = JitTrans(case=yml.get_case_info("LayerNorm_9"))
    jit_case.jit_run()
