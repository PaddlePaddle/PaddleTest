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


def test_Transformer_base():
    """test Transformer_base"""
    jit_case = JitTrans(case=yml.get_case_info("Transformer_base"))
    jit_case.jit_run()


def test_Transformer_0():
    """test Transformer_0"""
    jit_case = JitTrans(case=yml.get_case_info("Transformer_0"))
    jit_case.jit_run()


def test_Transformer_1():
    """test Transformer_1"""
    jit_case = JitTrans(case=yml.get_case_info("Transformer_1"))
    jit_case.jit_run()


def test_Transformer_2():
    """test Transformer_2"""
    jit_case = JitTrans(case=yml.get_case_info("Transformer_2"))
    jit_case.jit_run()


def test_Transformer_3():
    """test Transformer_3"""
    jit_case = JitTrans(case=yml.get_case_info("Transformer_3"))
    jit_case.jit_run()


def test_Transformer_4():
    """test Transformer_4"""
    jit_case = JitTrans(case=yml.get_case_info("Transformer_4"))
    jit_case.jit_run()


def test_Transformer_5():
    """test Transformer_5"""
    jit_case = JitTrans(case=yml.get_case_info("Transformer_5"))
    jit_case.jit_run()
