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


def test_Conv1DTranspose_base():
    """test Conv1DTranspose_base"""
    jit_case = JitTrans(case=yml.get_case_info("Conv1DTranspose_base"))
    jit_case.jit_run()


def test_Conv1DTranspose():
    """test Conv1DTranspose"""
    jit_case = JitTrans(case=yml.get_case_info("Conv1DTranspose"))
    jit_case.jit_run()


def test_Conv1DTranspose0():
    """test Conv1DTranspose0"""
    jit_case = JitTrans(case=yml.get_case_info("Conv1DTranspose0"))
    jit_case.jit_run()


def test_Conv1DTranspose1():
    """test Conv1DTranspose1"""
    jit_case = JitTrans(case=yml.get_case_info("Conv1DTranspose1"))
    jit_case.jit_run()


def test_Conv1DTranspose2():
    """test Conv1DTranspose2"""
    jit_case = JitTrans(case=yml.get_case_info("Conv1DTranspose2"))
    jit_case.jit_run()


def test_Conv1DTranspose3():
    """test Conv1DTranspose3"""
    jit_case = JitTrans(case=yml.get_case_info("Conv1DTranspose3"))
    jit_case.jit_run()


def test_Conv1DTranspose4():
    """test Conv1DTranspose4"""
    jit_case = JitTrans(case=yml.get_case_info("Conv1DTranspose4"))
    jit_case.jit_run()


def test_Conv1DTranspose5():
    """test Conv1DTranspose5"""
    jit_case = JitTrans(case=yml.get_case_info("Conv1DTranspose5"))
    jit_case.jit_run()


def test_Conv1DTranspose6():
    """test Conv1DTranspose6"""
    jit_case = JitTrans(case=yml.get_case_info("Conv1DTranspose6"))
    jit_case.jit_run()
