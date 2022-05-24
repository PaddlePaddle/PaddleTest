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


def test_SpectralNorm_base():
    """test SpectralNorm_base"""
    jit_case = JitTrans(case=yml.get_case_info("SpectralNorm_base"))
    jit_case.jit_run()


def test_SpectralNorm_0():
    """test SpectralNorm_0"""
    jit_case = JitTrans(case=yml.get_case_info("SpectralNorm_0"))
    jit_case.jit_run()


def test_SpectralNorm_1():
    """test SpectralNorm_1"""
    jit_case = JitTrans(case=yml.get_case_info("SpectralNorm_1"))
    jit_case.jit_run()


def test_SpectralNorm_2():
    """test SpectralNorm_2"""
    jit_case = JitTrans(case=yml.get_case_info("SpectralNorm_2"))
    jit_case.jit_run()


def test_SpectralNorm_3():
    """test SpectralNorm_3"""
    jit_case = JitTrans(case=yml.get_case_info("SpectralNorm_3"))
    jit_case.jit_run()


def test_SpectralNorm_4():
    """test SpectralNorm_4"""
    jit_case = JitTrans(case=yml.get_case_info("SpectralNorm_4"))
    jit_case.jit_run()


def test_SpectralNorm_5():
    """test SpectralNorm_5"""
    jit_case = JitTrans(case=yml.get_case_info("SpectralNorm_5"))
    jit_case.jit_run()


def test_SpectralNorm_6():
    """test SpectralNorm_6"""
    jit_case = JitTrans(case=yml.get_case_info("SpectralNorm_6"))
    jit_case.jit_run()


def test_SpectralNorm_7():
    """test SpectralNorm_7"""
    jit_case = JitTrans(case=yml.get_case_info("SpectralNorm_7"))
    jit_case.jit_run()


def test_SpectralNorm_9():
    """test SpectralNorm_9"""
    jit_case = JitTrans(case=yml.get_case_info("SpectralNorm_9"))
    jit_case.jit_run()
