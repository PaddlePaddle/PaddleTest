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


def test_BatchNorm1D_base():
    """test BatchNorm1D_base"""
    jit_case = JitTrans(case=yml.get_case_info("BatchNorm1D_base"))
    jit_case.jit_run()


def test_BatchNorm1D():
    """test BatchNorm1D"""
    jit_case = JitTrans(case=yml.get_case_info("BatchNorm1D"))
    jit_case.jit_run()


def test_BatchNorm1D2():
    """test BatchNorm1D2"""
    jit_case = JitTrans(case=yml.get_case_info("BatchNorm1D2"))
    jit_case.jit_run()


def test_BatchNorm1D3():
    """test BatchNorm1D3"""
    jit_case = JitTrans(case=yml.get_case_info("BatchNorm1D3"))
    jit_case.jit_run()


def test_BatchNorm1D4():
    """test BatchNorm1D4"""
    jit_case = JitTrans(case=yml.get_case_info("BatchNorm1D4"))
    jit_case.jit_run()


def test_BatchNorm1D5():
    """test BatchNorm1D5"""
    jit_case = JitTrans(case=yml.get_case_info("BatchNorm1D5"))
    jit_case.jit_run()


def test_BatchNorm1D6():
    """test BatchNorm1D6"""
    jit_case = JitTrans(case=yml.get_case_info("BatchNorm1D6"))
    jit_case.jit_run()


def test_BatchNorm1D7():
    """test BatchNorm1D7"""
    jit_case = JitTrans(case=yml.get_case_info("BatchNorm1D7"))
    jit_case.jit_run()


def test_BatchNorm1D8():
    """test BatchNorm1D8"""
    jit_case = JitTrans(case=yml.get_case_info("BatchNorm1D8"))
    jit_case.jit_run()


def test_BatchNorm1D9():
    """test BatchNorm1D9"""
    jit_case = JitTrans(case=yml.get_case_info("BatchNorm1D9"))
    jit_case.jit_run()
