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


def test_LocalResponseNorm_base():
    """test LocalResponseNorm_base"""
    jit_case = JitTrans(case=yml.get_case_info("LocalResponseNorm_base"))
    jit_case.jit_run()


def test_LocalResponseNorm():
    """test LocalResponseNorm"""
    jit_case = JitTrans(case=yml.get_case_info("LocalResponseNorm"))
    jit_case.jit_run()


def test_LocalResponseNorm0():
    """test LocalResponseNorm0"""
    jit_case = JitTrans(case=yml.get_case_info("LocalResponseNorm0"))
    jit_case.jit_run()


def test_LocalResponseNorm1():
    """test LocalResponseNorm1"""
    jit_case = JitTrans(case=yml.get_case_info("LocalResponseNorm1"))
    jit_case.jit_run()


def test_LocalResponseNorm2():
    """test LocalResponseNorm2"""
    jit_case = JitTrans(case=yml.get_case_info("LocalResponseNorm2"))
    jit_case.jit_run()


def test_LocalResponseNorm3():
    """test LocalResponseNorm3"""
    jit_case = JitTrans(case=yml.get_case_info("LocalResponseNorm3"))
    jit_case.jit_run()


def test_LocalResponseNorm4():
    """test LocalResponseNorm4"""
    jit_case = JitTrans(case=yml.get_case_info("LocalResponseNorm4"))
    jit_case.jit_run()
