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


def test_InstanceNorm2D():
    """test InstanceNorm2D"""
    jit_case = JitTrans(case=yml.get_case_info("InstanceNorm2D"))
    jit_case.jit_run()


def test_InstanceNorm2D2():
    """test InstanceNorm2D2"""
    jit_case = JitTrans(case=yml.get_case_info("InstanceNorm2D2"))
    jit_case.jit_run()


def test_InstanceNorm2D3():
    """test InstanceNorm2D3"""
    jit_case = JitTrans(case=yml.get_case_info("InstanceNorm2D3"))
    jit_case.jit_run()


def test_InstanceNorm2D4():
    """test InstanceNorm2D4"""
    jit_case = JitTrans(case=yml.get_case_info("InstanceNorm2D4"))
    jit_case.jit_run()


def test_InstanceNorm2D6():
    """test InstanceNorm2D6"""
    jit_case = JitTrans(case=yml.get_case_info("InstanceNorm2D6"))
    jit_case.jit_run()


def test_InstanceNorm2D8():
    """test InstanceNorm2D8"""
    jit_case = JitTrans(case=yml.get_case_info("InstanceNorm2D8"))
    jit_case.jit_run()


def test_InstanceNorm2D9():
    """test InstanceNorm2D9"""
    jit_case = JitTrans(case=yml.get_case_info("InstanceNorm2D9"))
    jit_case.jit_run()
