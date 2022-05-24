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


def test_SimpleRNNCell_base():
    """test SimpleRNNCell_base"""
    jit_case = JitTrans(case=yml.get_case_info("SimpleRNNCell_base"))
    jit_case.jit_run()


def test_SimpleRNNCell_0():
    """test SimpleRNNCell_0"""
    jit_case = JitTrans(case=yml.get_case_info("SimpleRNNCell_0"))
    jit_case.jit_run()


def test_SimpleRNNCell_1():
    """test SimpleRNNCell_1"""
    jit_case = JitTrans(case=yml.get_case_info("SimpleRNNCell_1"))
    jit_case.jit_run()


def test_SimpleRNNCell_2():
    """test SimpleRNNCell_2"""
    jit_case = JitTrans(case=yml.get_case_info("SimpleRNNCell_2"))
    jit_case.jit_run()


def test_SimpleRNNCell_3():
    """test SimpleRNNCell_3"""
    jit_case = JitTrans(case=yml.get_case_info("SimpleRNNCell_3"))
    jit_case.jit_run()
