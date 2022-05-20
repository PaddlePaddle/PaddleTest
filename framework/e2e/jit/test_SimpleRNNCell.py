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


def test_SimpleRNNCell():
    """test SimpleRNNCell"""
    jit_case = JitTrans(case=yml.get_case_info("SimpleRNNCell"))
    jit_case.jit_run()


def test_SimpleRNNCell0():
    """test SimpleRNNCell0"""
    jit_case = JitTrans(case=yml.get_case_info("SimpleRNNCell0"))
    jit_case.jit_run()


def test_SimpleRNNCell1():
    """test SimpleRNNCell1"""
    jit_case = JitTrans(case=yml.get_case_info("SimpleRNNCell1"))
    jit_case.jit_run()


def test_SimpleRNNCell2():
    """test SimpleRNNCell2"""
    jit_case = JitTrans(case=yml.get_case_info("SimpleRNNCell2"))
    jit_case.jit_run()
