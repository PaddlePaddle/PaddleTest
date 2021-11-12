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

import subprocess
import re
import pytest
import numpy as np

from RocmTestFramework import TestRecModel
from RocmTestFramework import RepoInitCustom
from RocmTestFramework import RepoRemove
from RocmTestFramework import RepoDataset
from RocmTestFramework import clean_process


def setup_module():
    """
    RepoInit
    """
    RepoInitCustom(repo="PaddleRec")


def teardown_module():
    """
    RepoRemove
    """
    RepoRemove(repo="PaddleRec")


def setup_function():
    """
    clean_process
    """
    clean_process()


def test_deepfm():
    """
    deepfm test case
    """
    model = TestRecModel(model="deepfm", directory="models/rank/deepfm")
    model.test_rec_train()


def test_ncf():
    """
    ncf test case
    """
    model = TestRecModel(model="ncf", directory="models/recall/ncf")
    model.test_rec_train()


def test_wide_deep():
    """
    wide_deep test case
    """
    model = TestRecModel(model="wide_deep", directory="models/rank/wide_deep")
    model.test_rec_train()


def test_word2vec():
    """
    word2vec test case
    """
    model = TestRecModel(model="word2vec", directory="models/recall/word2vec")
    model.test_rec_train()
