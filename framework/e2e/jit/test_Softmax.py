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


def test_LogSoftmax_base():
    """test LogSoftmax_base"""
    jit_case = JitTrans(case=yml.get_case_info("LogSoftmax_base"))
    jit_case.jit_run()


def test_LogSoftmax():
    """test LogSoftmax"""
    jit_case = JitTrans(case=yml.get_case_info("LogSoftmax"))
    jit_case.jit_run()


def test_LogSoftmax0():
    """test LogSoftmax0"""
    jit_case = JitTrans(case=yml.get_case_info("LogSoftmax0"))
    jit_case.jit_run()


def test_LogSoftmax1():
    """test LogSoftmax1"""
    jit_case = JitTrans(case=yml.get_case_info("LogSoftmax1"))
    jit_case.jit_run()


def test_LogSoftmax2():
    """test LogSoftmax2"""
    jit_case = JitTrans(case=yml.get_case_info("LogSoftmax2"))
    jit_case.jit_run()


def test_LogSoftmax3():
    """test LogSoftmax3"""
    jit_case = JitTrans(case=yml.get_case_info("LogSoftmax3"))
    jit_case.jit_run()


def test_LogSoftmax4():
    """test LogSoftmax4"""
    jit_case = JitTrans(case=yml.get_case_info("LogSoftmax4"))
    jit_case.jit_run()


def test_Softmax_base():
    """test Softmax_base"""
    jit_case = JitTrans(case=yml.get_case_info("Softmax_base"))
    jit_case.jit_run()


def test_Softmax():
    """test Softmax"""
    jit_case = JitTrans(case=yml.get_case_info("Softmax"))
    jit_case.jit_run()


def test_Softmax1():
    """test Softmax1"""
    jit_case = JitTrans(case=yml.get_case_info("Softmax1"))
    jit_case.jit_run()


def test_Softmax3():
    """test Softmax3"""
    jit_case = JitTrans(case=yml.get_case_info("Softmax3"))
    jit_case.jit_run()


def test_Softmax4():
    """test Softmax4"""
    jit_case = JitTrans(case=yml.get_case_info("Softmax4"))
    jit_case.jit_run()
