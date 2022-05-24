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


def test_InstanceNorm2D_base():
    """test InstanceNorm2D_base"""
    jit_case = JitTrans(case=yml.get_case_info("InstanceNorm2D_base"))
    jit_case.jit_run()


def test_InstanceNorm2D_0():
    """test InstanceNorm2D_0"""
    jit_case = JitTrans(case=yml.get_case_info("InstanceNorm2D_0"))
    jit_case.jit_run()


def test_InstanceNorm2D_2():
    """test InstanceNorm2D_2"""
    jit_case = JitTrans(case=yml.get_case_info("InstanceNorm2D_2"))
    jit_case.jit_run()


def test_InstanceNorm2D_3():
    """test InstanceNorm2D_3"""
    jit_case = JitTrans(case=yml.get_case_info("InstanceNorm2D_3"))
    jit_case.jit_run()


def test_InstanceNorm2D_4():
    """test InstanceNorm2D_4"""
    jit_case = JitTrans(case=yml.get_case_info("InstanceNorm2D_4"))
    jit_case.jit_run()


def test_InstanceNorm2D_6():
    """test InstanceNorm2D_6"""
    jit_case = JitTrans(case=yml.get_case_info("InstanceNorm2D_6"))
    jit_case.jit_run()


def test_InstanceNorm2D_8():
    """test InstanceNorm2D_8"""
    jit_case = JitTrans(case=yml.get_case_info("InstanceNorm2D_8"))
    jit_case.jit_run()


def test_InstanceNorm2D_9():
    """test InstanceNorm2D_9"""
    jit_case = JitTrans(case=yml.get_case_info("InstanceNorm2D_9"))
    jit_case.jit_run()
