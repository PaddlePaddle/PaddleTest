#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
/***************************************************************************
  *
  * Copyright (c) 2021 Baidu.com, Inc. All Rights Reserved
  * @file 
  * @author jiaxiao01
  * @date 2021/9/3 3:46 PM
  * @brief clas model inference test case
  *
  **************************************************************************/
"""

import pytest
import numpy as np
import subprocess
import re

from RocmTestFramework import TestParakeetModel
from RocmTestFramework import RepoInitCustom
from RocmTestFramework import RepoRemove
from RocmTestFramework import RepoDataset
from RocmTestFramework import clean_process


def setup_module():
    """
    """
    RepoInitCustom(repo='Parakeet')
    RepoDataset(cmd='''cd Parakeet; 
                       yum install -y libsndfile; 
                       python -m pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple;
                       rm -rf ~/nltk_data;
                       ln -s /data/nltk_data ~/nltk_data;''')

def teardown_module():
    """
    """
    RepoRemove(repo='Parakeet')


def setup_function():
    clean_process()


def test_transformer_tts():
    """
    transformer_tts test case
    """
    model = TestParakeetModel(model='transformer_tts')
    model.test_parakeet_train()

def test_waveflow():
    """
    waveflow test case
    """
    model = TestParakeetModel(model='waveflow')
    model.test_parakeet_train()

