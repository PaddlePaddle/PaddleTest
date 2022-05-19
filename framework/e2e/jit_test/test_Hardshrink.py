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


def test_Hardshrink_base():
    """test Hardshrink_base"""
    jit_case = JitTrans(case=yml.get_case_info("Hardshrink_base"))
    jit_case.jit_run()


def test_Hardshrink():
    """test Hardshrink"""
    jit_case = JitTrans(case=yml.get_case_info("Hardshrink"))
    jit_case.jit_run()


def test_Hardshrink1():
    """test Hardshrink1"""
    jit_case = JitTrans(case=yml.get_case_info("Hardshrink1"))
    jit_case.jit_run()


def test_Hardshrink2():
    """test Hardshrink2"""
    jit_case = JitTrans(case=yml.get_case_info("Hardshrink2"))
    jit_case.jit_run()


def test_Hardshrink3():
    """test Hardshrink3"""
    jit_case = JitTrans(case=yml.get_case_info("Hardshrink3"))
    jit_case.jit_run()
