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


def test_LeakyReLU_base():
    """test LeakyReLU_base"""
    jit_case = JitTrans(case=yml.get_case_info("LeakyReLU_base"))
    jit_case.jit_run()


def test_LeakyReLU():
    """test LeakyReLU"""
    jit_case = JitTrans(case=yml.get_case_info("LeakyReLU"))
    jit_case.jit_run()


def test_LeakyReLU1():
    """test LeakyReLU1"""
    jit_case = JitTrans(case=yml.get_case_info("LeakyReLU1"))
    jit_case.jit_run()


def test_LeakyReLU2():
    """test LeakyReLU2"""
    jit_case = JitTrans(case=yml.get_case_info("LeakyReLU2"))
    jit_case.jit_run()


def test_ReLU_base():
    """test ReLU_base"""
    jit_case = JitTrans(case=yml.get_case_info("ReLU_base"))
    jit_case.jit_run()


def test_ReLU():
    """test ReLU"""
    jit_case = JitTrans(case=yml.get_case_info("ReLU"))
    jit_case.jit_run()


def test_ReLU0():
    """test ReLU0"""
    jit_case = JitTrans(case=yml.get_case_info("ReLU0"))
    jit_case.jit_run()


def test_ReLU6_base():
    """test ReLU6_base"""
    jit_case = JitTrans(case=yml.get_case_info("ReLU6_base"))
    jit_case.jit_run()


def test_ReLU6():
    """test ReLU6"""
    jit_case = JitTrans(case=yml.get_case_info("ReLU6"))
    jit_case.jit_run()


def test_ReLU60():
    """test ReLU60"""
    jit_case = JitTrans(case=yml.get_case_info("ReLU60"))
    jit_case.jit_run()


def test_ReLU61():
    """test ReLU61"""
    jit_case = JitTrans(case=yml.get_case_info("ReLU61"))
    jit_case.jit_run()


def test_ThresholdedReLU_base():
    """test ThresholdedReLU_base"""
    jit_case = JitTrans(case=yml.get_case_info("ThresholdedReLU_base"))
    jit_case.jit_run()


def test_ThresholdedReLU():
    """test ThresholdedReLU"""
    jit_case = JitTrans(case=yml.get_case_info("ThresholdedReLU"))
    jit_case.jit_run()


def test_ThresholdedReLU1():
    """test ThresholdedReLU1"""
    jit_case = JitTrans(case=yml.get_case_info("ThresholdedReLU1"))
    jit_case.jit_run()


def test_ThresholdedReLU2():
    """test ThresholdedReLU2"""
    jit_case = JitTrans(case=yml.get_case_info("ThresholdedReLU2"))
    jit_case.jit_run()
