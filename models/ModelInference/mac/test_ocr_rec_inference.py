#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
/***************************************************************************
  *
  * Copyright (c) 2021 Baidu.com, Inc. All Rights Reserved
  * @file 
  * @author jiaxiao01
  * @date 2021/9/16 2:40 PM
  * @brief ocr rec model inference test case
  *
  **************************************************************************/
"""

import pytest
import numpy as np
import subprocess
import re

from ModelInfrenceFramework import TestOcrRecInference
from ModelInfrenceFramework import RepoInit
from ModelInfrenceFramework import clean_process


def setup_module():
    """
    git clone repo and install dependency
    """
    RepoInit(repo='PaddleOCR', branch='dygraph')

def setup_function():
    """
    clean process
    """
    clean_process()


def test_ch_PP_OCRv2_rec_infer():
    """
    test case
    """ 
    model = TestOcrRecInference(model='ch_PP-OCRv2_rec_infer', 
                                infer_imgs='doc/imgs_words/ch/word_1.jpg', 
                                rec_char_dict="./ppocr/utils/ppocr_keys_v1.txt")
    model.test_get_ocr_rec_inference_model(category='PP-OCRv2/chinese')
    model.test_ocr_rec_predict(expect_rec_docs='韩国小馆', expect_rec_scores=0.9967337)


def test_ch_PP_OCRv2_rec_slim_quant_infer():
    """
    test case
    """
    model = TestOcrRecInference(model='ch_PP-OCRv2_rec_slim_quant_infer', 
                                infer_imgs='doc/imgs_words/ch/word_1.jpg', 
                                rec_char_dict="./ppocr/utils/ppocr_keys_v1.txt")
    model.test_get_ocr_rec_inference_model(category='PP-OCRv2/chinese')
    model.test_ocr_rec_predict(expect_rec_docs='韩国小馆', expect_rec_scores=0.9967337)


def test_ch_ppocr_mobile_v2_rec_infer():
    """
    test case
    """
    model = TestOcrRecInference(model='ch_ppocr_mobile_v2.0_rec_infer', 
                                infer_imgs='doc/imgs_words/ch/word_1.jpg', 
                                rec_char_dict="./ppocr/utils/ppocr_keys_v1.txt")
    model.test_get_ocr_rec_inference_model(category='dygraph_v2.0/ch')
    model.test_ocr_rec_predict(expect_rec_docs='韩国小馆', expect_rec_scores=0.9967337)
    
def test_ch_ppocr_mobile_v2_rec_slim_infer():
    """
    test case
    """
    model = TestOcrRecInference(model='ch_ppocr_mobile_v2.0_rec_slim_infer',
                                infer_imgs='doc/imgs_words/ch/word_1.jpg',
                                rec_char_dict="./ppocr/utils/ppocr_keys_v1.txt")
    model.test_get_ocr_rec_inference_model(category='dygraph_v2.0/ch') 
    model.test_ocr_rec_predict(expect_rec_docs='韩国小馆', expect_rec_scores=0.9967337)

@pytest.mark.skip(reason='InvalidArgumentError: some trt inputs dynamic shape info not set')
def test_ch_ppocr_server_v2_rec_infer():
    """
    test case
    """
    model = TestOcrRecInference(model='ch_ppocr_server_v2.0_rec_infer', 
                                infer_imgs='doc/imgs_words/ch/word_1.jpg', 
                                rec_char_dict="./ppocr/utils/ppocr_keys_v1.txt")
    model.test_get_ocr_rec_inference_model(category='dygraph_v2.0/ch')
    model.test_ocr_rec_predict(expect_rec_docs='韩国小馆', expect_rec_scores=0.9967337)


def test_en_number_mobile_v2_rec_infer():
    """
    test case
    """
    model = TestOcrRecInference(model='en_number_mobile_v2.0_rec_infer', 
                                infer_imgs='doc/imgs_words/en/word_1.png', 
                                rec_char_dict="./ppocr/utils/en_dict.txt")
    model.test_get_ocr_rec_inference_model(category='dygraph_v2.0/multilingual')
    model.test_ocr_rec_predict(expect_rec_docs='JOINT', expect_rec_scores=0.8495678)

@pytest.mark.skip(reason='bug: predict result incorrect; rec_docs:]b\\ag,')
def test_en_number_mobile_v2_rec_slim_infer():
    """
    test case
    """
    model = TestOcrRecInference(model='en_number_mobile_v2.0_rec_slim_infer', 
                                infer_imgs='doc/imgs_words/en/word_1.png', 
                                rec_char_dict="./ppocr/utils/en_dict.txt")
    model.test_get_ocr_rec_inference_model(category='dygraph_v2.0/en')
    model.test_ocr_rec_predict(expect_rec_docs='JOINT', expect_rec_scores=0.8495678)


def test_french_mobile_v2_rec_infer():
    """
    test case
    """
    model = TestOcrRecInference(model='french_mobile_v2.0_rec_infer', 
                                infer_imgs='doc/imgs_words/french/2.jpg', 
                                rec_char_dict="ppocr/utils/dict/french_dict.txt")
    model.test_get_ocr_rec_inference_model(category='dygraph_v2.0/multilingual')
    model.test_ocr_rec_predict(expect_rec_docs='au développement.', expect_rec_scores=0.9999841)

@pytest.mark.skip(reason='bug: IndexError: list index out of range')
def test_german_mobile_v2_rec_infer():
    """
    test case
    """
    model = TestOcrRecInference(model='german_mobile_v2.0_rec_infer', 
                                infer_imgs='doc/imgs_words/german/1.jpg', 
                                rec_char_dict="ppocr/utils/dict/german_dict.txt")
    model.test_get_ocr_rec_inference_model(category='dygraph_v2.0/multilingual')
    model.test_ocr_rec_predict(expect_rec_docs='始終如壹', expect_rec_scores=0.9866849)

def test_korean_mobile_v2_rec_infer():
    """
    test case
    """
    model = TestOcrRecInference(model='korean_mobile_v2.0_rec_infer', 
                                infer_imgs='doc/imgs_words/korean/1.jpg', 
                                rec_char_dict="ppocr/utils/dict/korean_dict.txt")
    model.test_get_ocr_rec_inference_model(category='dygraph_v2.0/multilingual')
    model.test_ocr_rec_predict(expect_rec_docs='바탕으로', expect_rec_scores=0.98361814)

def test_japan_mobile_v2_rec_infer():
    """
    test case
    """
    model = TestOcrRecInference(model='japan_mobile_v2.0_rec_infer', 
                                infer_imgs='doc/imgs_words/japan/1.jpg', 
                                rec_char_dict="ppocr/utils/dict/japan_dict.txt")
    model.test_get_ocr_rec_inference_model(category='dygraph_v2.0/multilingual')
    model.test_ocr_rec_predict(expect_rec_docs='したがって', expect_rec_scores=0.9996532)


def test_chinese_cht_mobile_v2_rec_infer():
    """
    test case
    """
    model = TestOcrRecInference(model='chinese_cht_mobile_v2.0_rec_infer', 
                                infer_imgs='doc/imgs_words/chinese_traditional/chinese_cht_1.png', 
                                rec_char_dict="ppocr/utils/dict/chinese_cht_dict.txt")
    model.test_get_ocr_rec_inference_model(category='dygraph_v2.0/multilingual')
    model.test_ocr_rec_predict(expect_rec_docs='始終如壹', expect_rec_scores=0.9866849)

def test_te_mobile_v2_rec_infer():
    """
    test case
    """
    model = TestOcrRecInference(model='te_mobile_v2.0_rec_infer', 
                                infer_imgs='doc/imgs_words/telugu/te_1.jpg', 
                                rec_char_dict="ppocr/utils/dict/te_dict.txt")
    model.test_get_ocr_rec_inference_model(category='dygraph_v2.0/multilingual')
    model.test_ocr_rec_predict(expect_rec_docs='ఉంగరము', expect_rec_scores=0.99991435)

@pytest.mark.skip(reason='skip, :ೆೀಉಿಣಷಸ')
def test_ka_mobile_v2_rec_infer():
    """
    test case
    """
    model = TestOcrRecInference(model='ka_mobile_v2.0_rec_infer', 
                                infer_imgs='doc/imgs_words/kannada/ka_1.jpg', 
                                rec_char_dict="ppocr/utils/dict/ka_dict.txt")
    model.test_get_ocr_rec_inference_model(category='dygraph_v2.0/multilingual')
    model.test_ocr_rec_predict(expect_rec_docs=':ೆೀಉಿಣಷಸ', expect_rec_scores=0.9995285)

def test_ta_mobile_v2_rec_infer():
    """
    test case
    """
    model = TestOcrRecInference(model='ta_mobile_v2.0_rec_infer', 
                                infer_imgs='doc/imgs_words/tamil/ta_1.jpg', 
                                rec_char_dict="ppocr/utils/dict/ta_dict.txt")
    model.test_get_ocr_rec_inference_model(category='dygraph_v2.0/multilingual')
    model.test_ocr_rec_predict(expect_rec_docs='ஓி5னடிந', expect_rec_scores=0.9328843)

def test_latin_ppocr_mobile_v2_rec_infer():
    """
    test case
    """
    model = TestOcrRecInference(model='latin_ppocr_mobile_v2.0_rec_infer', 
                                infer_imgs='doc/imgs_words/french/2.jpg', 
                                rec_char_dict="ppocr/utils/dict/latin_dict.txt")
    model.test_get_ocr_rec_inference_model(category='dygraph_v2.0/multilingual')
    model.test_ocr_rec_predict(expect_rec_docs='audéveloppement.', expect_rec_scores=0.98615086)


def test_arabic_ppocr_mobile_v2_rec_infer():
    """
    test case
    """
    model = TestOcrRecInference(model='arabic_ppocr_mobile_v2.0_rec_infer', 
                                infer_imgs='doc/imgs_words/arabic/ar_1.jpg', 
                                rec_char_dict="ppocr/utils/dict/arabic_dict.txt")
    model.test_get_ocr_rec_inference_model(category='dygraph_v2.0/multilingual')
    model.test_ocr_rec_predict(expect_rec_docs='ل', expect_rec_scores=0.3066619)


def test_cyrillic_ppocr_mobile_v2_rec_infer():
    """
    test case
    """
    model = TestOcrRecInference(model='cyrillic_ppocr_mobile_v2.0_rec_infer', 
                                infer_imgs='doc/imgs_words/russia/ru_1.jpg', 
                                rec_char_dict="ppocr/utils/dict/cyrillic_dict.txt")
    model.test_get_ocr_rec_inference_model(category='dygraph_v2.0/multilingual')
    model.test_ocr_rec_predict(expect_rec_docs='разогна подманившие', expect_rec_scores=0.9999749)

def test_devanagari_ppocr_mobile_v2_rec_infer():
    """
    test case
    """
    model = TestOcrRecInference(model='devanagari_ppocr_mobile_v2.0_rec_infer', 
                                infer_imgs='doc/imgs_words/hindi/hi_1.jpg', 
                                rec_char_dict="ppocr/utils/dict/devanagari_dict.txt")
    model.test_get_ocr_rec_inference_model(category='dygraph_v2.0/multilingual')
    model.test_ocr_rec_predict(expect_rec_docs='वनस्पतियुक्त', expect_rec_scores=0.954812)

