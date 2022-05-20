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


def test_Conv3DTranspose():
    """test Conv3DTranspose"""
    jit_case = JitTrans(case=yml.get_case_info("Conv3DTranspose"))
    jit_case.jit_run()


def test_Conv3DTranspose0():
    """test Conv3DTranspose0"""
    jit_case = JitTrans(case=yml.get_case_info("Conv3DTranspose0"))
    jit_case.jit_run()


def test_Conv3DTranspose1():
    """test Conv3DTranspose1"""
    jit_case = JitTrans(case=yml.get_case_info("Conv3DTranspose1"))
    jit_case.jit_run()


def test_Conv3DTranspose2():
    """test Conv3DTranspose2"""
    jit_case = JitTrans(case=yml.get_case_info("Conv3DTranspose2"))
    jit_case.jit_run()


def test_Conv3DTranspose3():
    """test Conv3DTranspose3"""
    jit_case = JitTrans(case=yml.get_case_info("Conv3DTranspose3"))
    jit_case.jit_run()


def test_Conv3DTranspose4():
    """test Conv3DTranspose4"""
    jit_case = JitTrans(case=yml.get_case_info("Conv3DTranspose4"))
    jit_case.jit_run()


def test_Conv3DTranspose5():
    """test Conv3DTranspose5"""
    jit_case = JitTrans(case=yml.get_case_info("Conv3DTranspose5"))
    jit_case.jit_run()


def test_Conv3DTranspose6():
    """test Conv3DTranspose6"""
    jit_case = JitTrans(case=yml.get_case_info("Conv3DTranspose6"))
    jit_case.jit_run()
