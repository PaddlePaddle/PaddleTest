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


def test_UpsamplingNearest2D_base():
    """test UpsamplingNearest2D_base"""
    jit_case = JitTrans(case=yml.get_case_info("UpsamplingNearest2D_base"))
    jit_case.jit_run()


def test_UpsamplingNearest2D_0():
    """test UpsamplingNearest2D_0"""
    jit_case = JitTrans(case=yml.get_case_info("UpsamplingNearest2D_0"))
    jit_case.jit_run()


def test_UpsamplingNearest2D_1():
    """test UpsamplingNearest2D_1"""
    jit_case = JitTrans(case=yml.get_case_info("UpsamplingNearest2D_1"))
    jit_case.jit_run()


def test_UpsamplingNearest2D_2():
    """test UpsamplingNearest2D_2"""
    jit_case = JitTrans(case=yml.get_case_info("UpsamplingNearest2D_2"))
    jit_case.jit_run()


def test_UpsamplingNearest2D_3():
    """test UpsamplingNearest2D_3"""
    jit_case = JitTrans(case=yml.get_case_info("UpsamplingNearest2D_3"))
    jit_case.jit_run()


def test_UpsamplingNearest2D_4():
    """test UpsamplingNearest2D_4"""
    jit_case = JitTrans(case=yml.get_case_info("UpsamplingNearest2D_4"))
    jit_case.jit_run()


def test_UpsamplingNearest2D_5():
    """test UpsamplingNearest2D_5"""
    jit_case = JitTrans(case=yml.get_case_info("UpsamplingNearest2D_5"))
    jit_case.jit_run()


def test_UpsamplingNearest2D_6():
    """test UpsamplingNearest2D_6"""
    jit_case = JitTrans(case=yml.get_case_info("UpsamplingNearest2D_6"))
    jit_case.jit_run()


def test_UpsamplingNearest2D_7():
    """test UpsamplingNearest2D_7"""
    jit_case = JitTrans(case=yml.get_case_info("UpsamplingNearest2D_7"))
    jit_case.jit_run()


def test_UpsamplingNearest2D_8():
    """test UpsamplingNearest2D_8"""
    jit_case = JitTrans(case=yml.get_case_info("UpsamplingNearest2D_8"))
    jit_case.jit_run()
