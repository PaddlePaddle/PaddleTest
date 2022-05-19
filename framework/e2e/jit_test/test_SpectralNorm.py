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


def test_SpectralNorm():
    """test SpectralNorm"""
    jit_case = JitTrans(case=yml.get_case_info("SpectralNorm"))
    jit_case.jit_run()


def test_SpectralNorm0():
    """test SpectralNorm0"""
    jit_case = JitTrans(case=yml.get_case_info("SpectralNorm0"))
    jit_case.jit_run()


def test_SpectralNorm1():
    """test SpectralNorm1"""
    jit_case = JitTrans(case=yml.get_case_info("SpectralNorm1"))
    jit_case.jit_run()


def test_SpectralNorm2():
    """test SpectralNorm2"""
    jit_case = JitTrans(case=yml.get_case_info("SpectralNorm2"))
    jit_case.jit_run()


def test_SpectralNorm3():
    """test SpectralNorm3"""
    jit_case = JitTrans(case=yml.get_case_info("SpectralNorm3"))
    jit_case.jit_run()


def test_SpectralNorm4():
    """test SpectralNorm4"""
    jit_case = JitTrans(case=yml.get_case_info("SpectralNorm4"))
    jit_case.jit_run()


def test_SpectralNorm5():
    """test SpectralNorm5"""
    jit_case = JitTrans(case=yml.get_case_info("SpectralNorm5"))
    jit_case.jit_run()


def test_SpectralNorm6():
    """test SpectralNorm6"""
    jit_case = JitTrans(case=yml.get_case_info("SpectralNorm6"))
    jit_case.jit_run()


def test_SpectralNorm7():
    """test SpectralNorm7"""
    jit_case = JitTrans(case=yml.get_case_info("SpectralNorm7"))
    jit_case.jit_run()
