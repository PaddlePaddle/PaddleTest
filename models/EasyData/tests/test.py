#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test cylinder3d_steady cases
"""

import os
import sys
import pytest
from compare import compare

def test_0():
   """
   test PPDC1
   """
   compare("test_PPDC1.log","ClasOutput INFO","PPDC1_standard.txt")


def test_1():
   """
   test PPDC2
   """
   compare("test_PPDC1.log","ClasOutput INFO","PPDC1_standard.txt")


def test_2():
   """
   new ad
   """
   compare("test_PPDC1.log","ClasOutput INFO","PPDC1_standard.txt")


if __name__ == "__main__":
    code=pytest.main(["-sv", sys.argv[0]])
    sys.exit(code)

