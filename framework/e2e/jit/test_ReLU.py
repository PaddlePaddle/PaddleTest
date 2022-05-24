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


def test_LeakyReLU_0():
    """test LeakyReLU_0"""
    jit_case = JitTrans(case=yml.get_case_info("LeakyReLU_0"))
    jit_case.jit_run()


def test_LeakyReLU_1():
    """test LeakyReLU_1"""
    jit_case = JitTrans(case=yml.get_case_info("LeakyReLU_1"))
    jit_case.jit_run()


def test_LeakyReLU_2():
    """test LeakyReLU_2"""
    jit_case = JitTrans(case=yml.get_case_info("LeakyReLU_2"))
    jit_case.jit_run()


def test_ReLU_base():
    """test ReLU_base"""
    jit_case = JitTrans(case=yml.get_case_info("ReLU_base"))
    jit_case.jit_run()


def test_ReLU_0():
    """test ReLU_0"""
    jit_case = JitTrans(case=yml.get_case_info("ReLU_0"))
    jit_case.jit_run()


def test_ReLU_1():
    """test ReLU_1"""
    jit_case = JitTrans(case=yml.get_case_info("ReLU_1"))
    jit_case.jit_run()


def test_ThresholdedReLU_base():
    """test ThresholdedReLU_base"""
    jit_case = JitTrans(case=yml.get_case_info("ThresholdedReLU_base"))
    jit_case.jit_run()


def test_ThresholdedReLU_0():
    """test ThresholdedReLU_0"""
    jit_case = JitTrans(case=yml.get_case_info("ThresholdedReLU_0"))
    jit_case.jit_run()


def test_ThresholdedReLU_1():
    """test ThresholdedReLU_1"""
    jit_case = JitTrans(case=yml.get_case_info("ThresholdedReLU_1"))
    jit_case.jit_run()


def test_ThresholdedReLU_2():
    """test ThresholdedReLU_2"""
    jit_case = JitTrans(case=yml.get_case_info("ThresholdedReLU_2"))
    jit_case.jit_run()
