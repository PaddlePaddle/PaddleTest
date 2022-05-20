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


def test_UpsamplingNearest2D():
    """test UpsamplingNearest2D"""
    jit_case = JitTrans(case=yml.get_case_info("UpsamplingNearest2D"))
    jit_case.jit_run()


def test_UpsamplingNearest2D0():
    """test UpsamplingNearest2D0"""
    jit_case = JitTrans(case=yml.get_case_info("UpsamplingNearest2D0"))
    jit_case.jit_run()


def test_UpsamplingNearest2D1():
    """test UpsamplingNearest2D1"""
    jit_case = JitTrans(case=yml.get_case_info("UpsamplingNearest2D1"))
    jit_case.jit_run()


def test_UpsamplingNearest2D2():
    """test UpsamplingNearest2D2"""
    jit_case = JitTrans(case=yml.get_case_info("UpsamplingNearest2D2"))
    jit_case.jit_run()


def test_UpsamplingNearest2D3():
    """test UpsamplingNearest2D3"""
    jit_case = JitTrans(case=yml.get_case_info("UpsamplingNearest2D3"))
    jit_case.jit_run()


def test_UpsamplingNearest2D4():
    """test UpsamplingNearest2D4"""
    jit_case = JitTrans(case=yml.get_case_info("UpsamplingNearest2D4"))
    jit_case.jit_run()


def test_UpsamplingNearest2D5():
    """test UpsamplingNearest2D5"""
    jit_case = JitTrans(case=yml.get_case_info("UpsamplingNearest2D5"))
    jit_case.jit_run()


def test_UpsamplingNearest2D6():
    """test UpsamplingNearest2D6"""
    jit_case = JitTrans(case=yml.get_case_info("UpsamplingNearest2D6"))
    jit_case.jit_run()


def test_UpsamplingNearest2D7():
    """test UpsamplingNearest2D7"""
    jit_case = JitTrans(case=yml.get_case_info("UpsamplingNearest2D7"))
    jit_case.jit_run()
