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


def test_Conv3DTranspose_base():
    """test Conv3DTranspose_base"""
    jit_case = JitTrans(case=yml.get_case_info("Conv3DTranspose_base"))
    jit_case.jit_run()


def test_Conv3DTranspose_0():
    """test Conv3DTranspose_0"""
    jit_case = JitTrans(case=yml.get_case_info("Conv3DTranspose_0"))
    jit_case.jit_run()


def test_Conv3DTranspose_1():
    """test Conv3DTranspose_1"""
    jit_case = JitTrans(case=yml.get_case_info("Conv3DTranspose_1"))
    jit_case.jit_run()


def test_Conv3DTranspose_2():
    """test Conv3DTranspose_2"""
    jit_case = JitTrans(case=yml.get_case_info("Conv3DTranspose_2"))
    jit_case.jit_run()


def test_Conv3DTranspose_3():
    """test Conv3DTranspose_3"""
    jit_case = JitTrans(case=yml.get_case_info("Conv3DTranspose_3"))
    jit_case.jit_run()


def test_Conv3DTranspose_4():
    """test Conv3DTranspose_4"""
    jit_case = JitTrans(case=yml.get_case_info("Conv3DTranspose_4"))
    jit_case.jit_run()


def test_Conv3DTranspose_5():
    """test Conv3DTranspose_5"""
    jit_case = JitTrans(case=yml.get_case_info("Conv3DTranspose_5"))
    jit_case.jit_run()


def test_Conv3DTranspose_6():
    """test Conv3DTranspose_6"""
    jit_case = JitTrans(case=yml.get_case_info("Conv3DTranspose_6"))
    jit_case.jit_run()


def test_Conv3DTranspose_7():
    """test Conv3DTranspose_7"""
    jit_case = JitTrans(case=yml.get_case_info("Conv3DTranspose_7"))
    jit_case.jit_run()
