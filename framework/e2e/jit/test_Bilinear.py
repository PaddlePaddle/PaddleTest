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


def test_Bilinear_base():
    """test Bilinear_base"""
    jit_case = JitTrans(case=yml.get_case_info("Bilinear_base"))
    jit_case.jit_run()


def test_Bilinear():
    """test Bilinear"""
    jit_case = JitTrans(case=yml.get_case_info("Bilinear"))
    jit_case.jit_run()


def test_UpsamplingBilinear2D_base():
    """test UpsamplingBilinear2D_base"""
    jit_case = JitTrans(case=yml.get_case_info("UpsamplingBilinear2D_base"))
    jit_case.jit_run()


def test_UpsamplingBilinear2D():
    """test UpsamplingBilinear2D"""
    jit_case = JitTrans(case=yml.get_case_info("UpsamplingBilinear2D"))
    jit_case.jit_run()


def test_UpsamplingBilinear2D1():
    """test UpsamplingBilinear2D1"""
    jit_case = JitTrans(case=yml.get_case_info("UpsamplingBilinear2D1"))
    jit_case.jit_run()


def test_UpsamplingBilinear2D2():
    """test UpsamplingBilinear2D2"""
    jit_case = JitTrans(case=yml.get_case_info("UpsamplingBilinear2D2"))
    jit_case.jit_run()


def test_UpsamplingBilinear2D3():
    """test UpsamplingBilinear2D3"""
    jit_case = JitTrans(case=yml.get_case_info("UpsamplingBilinear2D3"))
    jit_case.jit_run()


def test_UpsamplingBilinear2D4():
    """test UpsamplingBilinear2D4"""
    jit_case = JitTrans(case=yml.get_case_info("UpsamplingBilinear2D4"))
    jit_case.jit_run()


def test_UpsamplingBilinear2D5():
    """test UpsamplingBilinear2D5"""
    jit_case = JitTrans(case=yml.get_case_info("UpsamplingBilinear2D5"))
    jit_case.jit_run()


def test_UpsamplingBilinear2D6():
    """test UpsamplingBilinear2D6"""
    jit_case = JitTrans(case=yml.get_case_info("UpsamplingBilinear2D6"))
    jit_case.jit_run()


def test_UpsamplingBilinear2D7():
    """test UpsamplingBilinear2D7"""
    jit_case = JitTrans(case=yml.get_case_info("UpsamplingBilinear2D7"))
    jit_case.jit_run()
