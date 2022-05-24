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


def test_CosineSimilarity_base():
    """test CosineSimilarity_base"""
    jit_case = JitTrans(case=yml.get_case_info("CosineSimilarity_base"))
    jit_case.jit_run()


def test_CosineSimilarity_0():
    """test CosineSimilarity_0"""
    jit_case = JitTrans(case=yml.get_case_info("CosineSimilarity_0"))
    jit_case.jit_run()


def test_CosineSimilarity_1():
    """test CosineSimilarity_1"""
    jit_case = JitTrans(case=yml.get_case_info("CosineSimilarity_1"))
    jit_case.jit_run()


def test_CosineSimilarity_2():
    """test CosineSimilarity_2"""
    jit_case = JitTrans(case=yml.get_case_info("CosineSimilarity_2"))
    jit_case.jit_run()


def test_CosineSimilarity_3():
    """test CosineSimilarity_3"""
    jit_case = JitTrans(case=yml.get_case_info("CosineSimilarity_3"))
    jit_case.jit_run()


def test_CosineSimilarity_4():
    """test CosineSimilarity_4"""
    jit_case = JitTrans(case=yml.get_case_info("CosineSimilarity_4"))
    jit_case.jit_run()


def test_CosineSimilarity_5():
    """test CosineSimilarity_5"""
    jit_case = JitTrans(case=yml.get_case_info("CosineSimilarity_5"))
    jit_case.jit_run()
