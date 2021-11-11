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

import subprocess
import re
import pytest
import numpy as np

from ModelInfrenceFramework import TestOcrRecInference
from ModelInfrenceFramework import RepoInit
from ModelInfrenceFramework import clean_process
from ModelInfrenceFramework import RepoInstructions


def setup_module():
    """
    git clone repo and install dependency
    """
    RepoInit(repo="PaddleOCR", branch="dygraph")
    RepoInstructions(
        cmd="""
sed -i '/config.enable_tensorrt_engine/i\\            config.collect_shape_range_info("shape_range.pbtxt")'  \
tools/infer/utility.py; \
sed -i '/min_subgraph_size=args.min_subgraph_size)/a\\
            config.enable_tuned_tensorrt_dynamic_shape("shape_range.pbtxt", True)'  tools/infer/utility.py"""
    )


def setup_function():
    """
    clean process
    """
    clean_process()


def test_rec_r34_vd_none_none_ctc():
    """
    test case
    """
    model = TestOcrRecInference(
        model="rec_r34_vd_none_none_ctc_v2.0",
        infer_imgs="./doc/imgs_words_en/word_336.png",
        rec_char_dict="./ppocr/utils/ic15_dict.txt",
        yaml="configs/rec/rec_r34_vd_none_none_ctc.yml",
        algorithm="Rosetta",
    )

    model.test_get_ocr_rec_train_model(category="dygraph_v2.0/en")
    model.test_ocr_rec_export_model()
    model.test_ocr_rec_predict(expect_rec_docs="super", expect_rec_scores=0.99604285)


def test_rec_mv3_none_none_ctc():
    """
    test case
    """
    model = TestOcrRecInference(
        model="rec_mv3_none_none_ctc_v2.0",
        infer_imgs="./doc/imgs_words_en/word_336.png",
        rec_char_dict="./ppocr/utils/ic15_dict.txt",
        yaml="configs/rec/rec_mv3_none_none_ctc.yml",
        algorithm="Rosetta",
    )

    model.test_get_ocr_rec_train_model(category="dygraph_v2.0/en")
    model.test_ocr_rec_export_model()
    model.test_ocr_rec_predict(expect_rec_docs="super", expect_rec_scores=0.99604285)


def test_rec_r34_vd_none_bilstm_ctc():
    """
    test case
    """
    model = TestOcrRecInference(
        model="rec_r34_vd_none_bilstm_ctc_v2.0",
        infer_imgs="./doc/imgs_words_en/word_336.png",
        rec_char_dict="./ppocr/utils/ic15_dict.txt",
        yaml="configs/rec/rec_r34_vd_none_bilstm_ctc.yml",
        algorithm="CRNN",
    )

    model.test_get_ocr_rec_train_model(category="dygraph_v2.0/en")
    model.test_ocr_rec_export_model()
    model.test_ocr_rec_predict(expect_rec_docs="super", expect_rec_scores=0.99604285)


def test_rec_mv3_none_bilstm_ctc():
    """
    test case
    """
    model = TestOcrRecInference(
        model="rec_mv3_none_bilstm_ctc_v2.0",
        infer_imgs="./doc/imgs_words_en/word_336.png",
        rec_char_dict="./ppocr/utils/ic15_dict.txt",
        yaml="configs/rec/rec_mv3_none_bilstm_ctc.yml",
        algorithm="CRNN",
    )

    model.test_get_ocr_rec_train_model(category="dygraph_v2.0/en")
    model.test_ocr_rec_export_model()
    model.test_ocr_rec_predict(expect_rec_docs="super", expect_rec_scores=0.99604285)


def test_rec_r34_vd_tps_bilstm_ctc():
    """
    rec_r34_vd_tps_bilstm_ctc test case
    """
    model = TestOcrRecInference(
        model="rec_r34_vd_tps_bilstm_ctc_v2.0",
        infer_imgs="./doc/imgs_words_en/word_336.png",
        rec_char_dict="./ppocr/utils/ic15_dict.txt",
        yaml="configs/rec/rec_r34_vd_tps_bilstm_ctc.yml",
        algorithm="StarNet",
    )

    model.test_get_ocr_rec_train_model(category="dygraph_v2.0/en")
    model.test_ocr_rec_export_model()
    model.test_ocr_rec_predict(expect_rec_docs="super", expect_rec_scores=0.9999531)


def test_rec_mv3_tps_bilstm_ctc():
    """
    rec_mv3_tps_bilstm_ctc test case
    """
    model = TestOcrRecInference(
        model="rec_mv3_tps_bilstm_ctc_v2.0",
        infer_imgs="./doc/imgs_words_en/word_336.png",
        rec_char_dict="./ppocr/utils/ic15_dict.txt",
        yaml="configs/rec/rec_mv3_tps_bilstm_ctc.yml",
        algorithm="StarNet",
    )

    model.test_get_ocr_rec_train_model(category="dygraph_v2.0/en")
    model.test_ocr_rec_export_model()
    model.test_ocr_rec_predict(expect_rec_docs="super", expect_rec_scores=0.9999531)


def test_rec_r34_vd_tps_bilstm_att():
    """
    test case
    """
    model = TestOcrRecInference(
        model="rec_r34_vd_tps_bilstm_att_v2.0",
        infer_imgs="./doc/imgs_words_en/word_336.png",
        rec_char_dict="./ppocr/utils/ic15_dict.txt",
        yaml="configs/rec/rec_r34_vd_tps_bilstm_att.yml",
        algorithm="RARE",
    )

    model.test_get_ocr_rec_train_model(category="dygraph_v2.0/en")
    model.test_ocr_rec_export_model()
    model.test_ocr_rec_predict(expect_rec_docs="super", expect_rec_scores=16.825443)


def test_rec_mv3_tps_bilstm_att():
    """
    test case
    """
    model = TestOcrRecInference(
        model="rec_mv3_tps_bilstm_att_v2.0",
        infer_imgs="./doc/imgs_words_en/word_336.png",
        rec_char_dict="./ppocr/utils/ic15_dict.txt",
        yaml="configs/rec/rec_mv3_tps_bilstm_att.yml",
        algorithm="RARE",
    )

    model.test_get_ocr_rec_train_model(category="dygraph_v2.0/en")
    model.test_ocr_rec_export_model()
    model.test_ocr_rec_predict(expect_rec_docs="super", expect_rec_scores=16.957952)


def test_rec_r50_vd_srn():
    """
    test case
    """
    model = TestOcrRecInference(
        model="rec_r50_vd_srn",
        infer_imgs="./doc/imgs_words_en/word_336.png",
        rec_char_dict="./ppocr/utils/ic15_dict.txt",
        yaml="configs/rec/rec_r50_fpn_srn.yml",
        algorithm="SRN",
        rec_image_shape="1,64,256",
    )

    model.test_get_ocr_rec_train_model(category="dygraph_v2.0/en")
    model.test_ocr_rec_export_model()
    model.test_ocr_rec_predict(expect_rec_docs="super", expect_rec_scores=0.9999999)


def test_rec_r31_sar():
    """
    test case
    """
    model = TestOcrRecInference(
        model="rec_r31_sar",
        infer_imgs="./doc/imgs_words/en/word_1.png",
        rec_char_dict="ppocr/utils/dict90.txt",
        yaml="configs/rec/rec_r31_sar.yml",
        algorithm="SAR",
        rec_image_shape="3,48,48,160",
    )

    model.test_get_ocr_rec_train_model(category="dygraph_v2.1/rec")
    model.test_ocr_rec_export_model()
    model.test_ocr_rec_predict(expect_rec_docs="JOINT", expect_rec_scores=0.99992895)


def test_rec_mtb_nrtr():
    """
    test case
    """
    model = TestOcrRecInference(
        model="rec_mtb_nrtr",
        infer_imgs="./doc/imgs_words_en/word_10.png",
        rec_char_dict="./ppocr/utils/EN_symbol_dict.txt",
        yaml="configs/rec/rec_mtb_nrtr.yml",
        algorithm="NRTR",
        rec_image_shape="1,32,100",
    )

    model.test_get_ocr_rec_train_model(category="dygraph_v2.0/en")
    model.test_ocr_rec_export_model()
    model.test_ocr_rec_predict(expect_rec_docs="pain", expect_rec_scores=0.92658794)


def test_rec_resnet_stn_bilstm_att():
    """
    test case
    """
    model = TestOcrRecInference(
        model="rec_resnet_stn_bilstm_att",
        infer_imgs="./doc/imgs_words_en/word_336.png",
        rec_char_dict="/ppocr/utils/ic15_dict.txt",
        yaml="configs/rec/rec_resnet_stn_bilstm_att.yml",
        algorithm="SEED",
        rec_image_shape="3,32,100",
    )

    model.test_get_ocr_rec_train_model(category="dygraph_v2.1/rec")
    model.test_ocr_rec_export_model()
    model.test_ocr_rec_predict(expect_rec_docs="super", expect_rec_scores=0.9999999)
