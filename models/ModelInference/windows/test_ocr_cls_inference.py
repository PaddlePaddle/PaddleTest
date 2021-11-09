#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
/***************************************************************************
  *
  * Copyright (c) 2021 Baidu.com, Inc. All Rights Reserved
  * @file 
  * @author jiaxiao01
  * @date 2021/9/10 3:46 PM
  * @brief clas model inference test case
  *
  **************************************************************************/
"""

import pytest
import numpy as np
import subprocess
import re


from ModelInfrenceFramework import TestOcrClsInference
from ModelInfrenceFramework import RepoInit
from ModelInfrenceFramework import clean_process


def setup_module():
    """
    git clone repo and install dependency
    """
    RepoInit(repo='PaddleOCR', branch='dygraph')

   

def test_ch_ppocr_mobile_v2_cls_infer():
    """
    test case
    """ 
    model = TestOcrClsInference(model='ch_ppocr_mobile_v2.0_cls_infer', 
                                infer_imgs='./doc/imgs_words/ch/word_5.jpg', 
                                cls_char_dict="./ppocr/utils/ppocr_keys_v1.txt")
    model.test_get_ocr_cls_inference_model(category='dygraph_v2.0/ch')
    model.test_ocr_cls_predict(expect_cls_docs='0', expect_cls_scores=0.9999988)

def test_ch_ppocr_mobile_v2_cls_slim_infer():
    """
    test case
    """
    model = TestOcrClsInference(model='ch_ppocr_mobile_v2.0_cls_slim_infer', 
                                infer_imgs='./doc/imgs_words/ch/word_5.jpg', 
                                cls_char_dict="./ppocr/utils/ppocr_keys_v1.txt")
    model.test_get_ocr_cls_inference_model(category='dygraph_v2.0/ch')
    model.test_ocr_cls_predict(expect_cls_docs='0', expect_cls_scores=0.9999988)
