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


def test_PixelShuffle_base():
    """test PixelShuffle_base"""
    jit_case = JitTrans(case=yml.get_case_info("PixelShuffle_base"))
    jit_case.jit_run()


def test_PixelShuffle_0():
    """test PixelShuffle_0"""
    jit_case = JitTrans(case=yml.get_case_info("PixelShuffle_0"))
    jit_case.jit_run()


def test_PixelShuffle_1():
    """test PixelShuffle_1"""
    jit_case = JitTrans(case=yml.get_case_info("PixelShuffle_1"))
    jit_case.jit_run()


def test_PixelShuffle_2():
    """test PixelShuffle_2"""
    jit_case = JitTrans(case=yml.get_case_info("PixelShuffle_2"))
    jit_case.jit_run()


def test_PixelShuffle_4():
    """test PixelShuffle_4"""
    jit_case = JitTrans(case=yml.get_case_info("PixelShuffle_4"))
    jit_case.jit_run()


def test_PixelShuffle_5():
    """test PixelShuffle_5"""
    jit_case = JitTrans(case=yml.get_case_info("PixelShuffle_5"))
    jit_case.jit_run()
