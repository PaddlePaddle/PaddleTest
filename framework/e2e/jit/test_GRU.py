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


def test_GRU_base():
    """test GRU_base"""
    jit_case = JitTrans(case=yml.get_case_info("GRU_base"))
    jit_case.jit_run()


def test_GRU():
    """test GRU"""
    jit_case = JitTrans(case=yml.get_case_info("GRU"))
    jit_case.jit_run()


def test_GRU1():
    """test GRU1"""
    jit_case = JitTrans(case=yml.get_case_info("GRU1"))
    jit_case.jit_run()


def test_GRU2():
    """test GRU2"""
    jit_case = JitTrans(case=yml.get_case_info("GRU2"))
    jit_case.jit_run()


def test_GRU3():
    """test GRU3"""
    jit_case = JitTrans(case=yml.get_case_info("GRU3"))
    jit_case.jit_run()


def test_GRU5():
    """test GRU5"""
    jit_case = JitTrans(case=yml.get_case_info("GRU5"))
    jit_case.jit_run()


def test_GRU6():
    """test GRU6"""
    jit_case = JitTrans(case=yml.get_case_info("GRU6"))
    jit_case.jit_run()
