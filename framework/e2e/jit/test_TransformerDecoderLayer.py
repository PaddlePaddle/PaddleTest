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


def test_TransformerDecoderLayer_base():
    """test TransformerDecoderLayer_base"""
    jit_case = JitTrans(case=yml.get_case_info("TransformerDecoderLayer_base"))
    jit_case.jit_run()


def test_TransformerDecoderLayer_0():
    """test TransformerDecoderLayer_0"""
    jit_case = JitTrans(case=yml.get_case_info("TransformerDecoderLayer_0"))
    jit_case.jit_run()


def test_TransformerDecoderLayer_1():
    """test TransformerDecoderLayer_1"""
    jit_case = JitTrans(case=yml.get_case_info("TransformerDecoderLayer_1"))
    jit_case.jit_run()


def test_TransformerDecoderLayer_2():
    """test TransformerDecoderLayer_2"""
    jit_case = JitTrans(case=yml.get_case_info("TransformerDecoderLayer_2"))
    jit_case.jit_run()


def test_TransformerDecoderLayer_3():
    """test TransformerDecoderLayer_3"""
    jit_case = JitTrans(case=yml.get_case_info("TransformerDecoderLayer_3"))
    jit_case.jit_run()


def test_TransformerDecoderLayer_4():
    """test TransformerDecoderLayer_4"""
    jit_case = JitTrans(case=yml.get_case_info("TransformerDecoderLayer_4"))
    jit_case.jit_run()
