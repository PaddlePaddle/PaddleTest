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


def test_TransformerEncoderLayer_base():
    """test TransformerEncoderLayer_base"""
    jit_case = JitTrans(case=yml.get_case_info("TransformerEncoderLayer_base"))
    jit_case.jit_run()


def test_TransformerEncoderLayer():
    """test TransformerEncoderLayer"""
    jit_case = JitTrans(case=yml.get_case_info("TransformerEncoderLayer"))
    jit_case.jit_run()


def test_TransformerEncoderLayer0():
    """test TransformerEncoderLayer0"""
    jit_case = JitTrans(case=yml.get_case_info("TransformerEncoderLayer0"))
    jit_case.jit_run()


def test_TransformerEncoderLayer1():
    """test TransformerEncoderLayer1"""
    jit_case = JitTrans(case=yml.get_case_info("TransformerEncoderLayer1"))
    jit_case.jit_run()


def test_TransformerEncoderLayer2():
    """test TransformerEncoderLayer2"""
    jit_case = JitTrans(case=yml.get_case_info("TransformerEncoderLayer2"))
    jit_case.jit_run()


def test_TransformerEncoderLayer3():
    """test TransformerEncoderLayer3"""
    jit_case = JitTrans(case=yml.get_case_info("TransformerEncoderLayer3"))
    jit_case.jit_run()
