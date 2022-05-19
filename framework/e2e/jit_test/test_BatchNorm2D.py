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


def test_BatchNorm2D_base():
    """test BatchNorm2D_base"""
    jit_case = JitTrans(case=yml.get_case_info("BatchNorm2D_base"))
    jit_case.jit_run()


def test_BatchNorm2D1():
    """test BatchNorm2D1"""
    jit_case = JitTrans(case=yml.get_case_info("BatchNorm2D1"))
    jit_case.jit_run()


def test_BatchNorm2D2():
    """test BatchNorm2D2"""
    jit_case = JitTrans(case=yml.get_case_info("BatchNorm2D2"))
    jit_case.jit_run()


def test_BatchNorm2D3():
    """test BatchNorm2D3"""
    jit_case = JitTrans(case=yml.get_case_info("BatchNorm2D3"))
    jit_case.jit_run()


def test_BatchNorm2D4():
    """test BatchNorm2D4"""
    jit_case = JitTrans(case=yml.get_case_info("BatchNorm2D4"))
    jit_case.jit_run()


def test_BatchNorm2D7():
    """test BatchNorm2D7"""
    jit_case = JitTrans(case=yml.get_case_info("BatchNorm2D7"))
    jit_case.jit_run()


def test_BatchNorm2D8():
    """test BatchNorm2D8"""
    jit_case = JitTrans(case=yml.get_case_info("BatchNorm2D8"))
    jit_case.jit_run()
