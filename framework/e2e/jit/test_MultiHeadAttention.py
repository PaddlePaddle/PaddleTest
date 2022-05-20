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


def test_MultiHeadAttention_base():
    """test MultiHeadAttention_base"""
    jit_case = JitTrans(case=yml.get_case_info("MultiHeadAttention_base"))
    jit_case.jit_run()


def test_MultiHeadAttention():
    """test MultiHeadAttention"""
    jit_case = JitTrans(case=yml.get_case_info("MultiHeadAttention"))
    jit_case.jit_run()


def test_MultiHeadAttention0():
    """test MultiHeadAttention0"""
    jit_case = JitTrans(case=yml.get_case_info("MultiHeadAttention0"))
    jit_case.jit_run()


def test_MultiHeadAttention1():
    """test MultiHeadAttention1"""
    jit_case = JitTrans(case=yml.get_case_info("MultiHeadAttention1"))
    jit_case.jit_run()


def test_MultiHeadAttention2():
    """test MultiHeadAttention2"""
    jit_case = JitTrans(case=yml.get_case_info("MultiHeadAttention2"))
    jit_case.jit_run()


def test_MultiHeadAttention3():
    """test MultiHeadAttention3"""
    jit_case = JitTrans(case=yml.get_case_info("MultiHeadAttention3"))
    jit_case.jit_run()
