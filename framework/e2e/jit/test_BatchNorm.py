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


def test_BatchNorm_base():
    """test BatchNorm_base"""
    jit_case = JitTrans(case=yml.get_case_info("BatchNorm_base"))
    jit_case.jit_run()


def test_BatchNorm_0():
    """test BatchNorm_0"""
    jit_case = JitTrans(case=yml.get_case_info("BatchNorm_0"))
    jit_case.jit_run()


def test_BatchNorm_1():
    """test BatchNorm_1"""
    jit_case = JitTrans(case=yml.get_case_info("BatchNorm_1"))
    jit_case.jit_run()


def test_BatchNorm_2():
    """test BatchNorm_2"""
    jit_case = JitTrans(case=yml.get_case_info("BatchNorm_2"))
    jit_case.jit_run()


def test_BatchNorm_3():
    """test BatchNorm_3"""
    jit_case = JitTrans(case=yml.get_case_info("BatchNorm_3"))
    jit_case.jit_run()


def test_BatchNorm_4():
    """test BatchNorm_4"""
    jit_case = JitTrans(case=yml.get_case_info("BatchNorm_4"))
    jit_case.jit_run()


def test_BatchNorm_7():
    """test BatchNorm_7"""
    jit_case = JitTrans(case=yml.get_case_info("BatchNorm_7"))
    jit_case.jit_run()


def test_BatchNorm_8():
    """test BatchNorm_8"""
    jit_case = JitTrans(case=yml.get_case_info("BatchNorm_8"))
    jit_case.jit_run()


def test_BatchNorm_10():
    """test BatchNorm_10"""
    jit_case = JitTrans(case=yml.get_case_info("BatchNorm_10"))
    jit_case.jit_run()


def test_BatchNorm_11():
    """test BatchNorm_11"""
    jit_case = JitTrans(case=yml.get_case_info("BatchNorm_11"))
    jit_case.jit_run()


def test_BatchNorm_12():
    """test BatchNorm_12"""
    jit_case = JitTrans(case=yml.get_case_info("BatchNorm_12"))
    jit_case.jit_run()


def test_BatchNorm_13():
    """test BatchNorm_13"""
    jit_case = JitTrans(case=yml.get_case_info("BatchNorm_13"))
    jit_case.jit_run()


def test_BatchNorm_15():
    """test BatchNorm_15"""
    jit_case = JitTrans(case=yml.get_case_info("BatchNorm_15"))
    jit_case.jit_run()


def test_BatchNorm_17():
    """test BatchNorm_17"""
    jit_case = JitTrans(case=yml.get_case_info("BatchNorm_17"))
    jit_case.jit_run()


def test_BatchNorm_18():
    """test BatchNorm_18"""
    jit_case = JitTrans(case=yml.get_case_info("BatchNorm_18"))
    jit_case.jit_run()
