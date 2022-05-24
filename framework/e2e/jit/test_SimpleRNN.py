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


def test_SimpleRNN_base():
    """test SimpleRNN_base"""
    jit_case = JitTrans(case=yml.get_case_info("SimpleRNN_base"))
    jit_case.jit_run()


def test_SimpleRNN_0():
    """test SimpleRNN_0"""
    jit_case = JitTrans(case=yml.get_case_info("SimpleRNN_0"))
    jit_case.jit_run()


def test_SimpleRNN_1():
    """test SimpleRNN_1"""
    jit_case = JitTrans(case=yml.get_case_info("SimpleRNN_1"))
    jit_case.jit_run()


def test_SimpleRNN_2():
    """test SimpleRNN_2"""
    jit_case = JitTrans(case=yml.get_case_info("SimpleRNN_2"))
    jit_case.jit_run()


def test_SimpleRNN_3():
    """test SimpleRNN_3"""
    jit_case = JitTrans(case=yml.get_case_info("SimpleRNN_3"))
    jit_case.jit_run()


def test_SimpleRNN_4():
    """test SimpleRNN_4"""
    jit_case = JitTrans(case=yml.get_case_info("SimpleRNN_4"))
    jit_case.jit_run()


def test_SimpleRNN_5():
    """test SimpleRNN_5"""
    jit_case = JitTrans(case=yml.get_case_info("SimpleRNN_5"))
    jit_case.jit_run()


def test_SimpleRNN_6():
    """test SimpleRNN_6"""
    jit_case = JitTrans(case=yml.get_case_info("SimpleRNN_6"))
    jit_case.jit_run()
