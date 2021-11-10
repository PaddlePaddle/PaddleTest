#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
/***************************************************************************
  *
  * Copyright (c) 2021 Baidu.com, Inc. All Rights Reserved
  * @file apibase.py
  * @author jiaxiao01
  * @date 2021/9/10 3:46 PM
  * @brief clas model inference test case
  *
  **************************************************************************/
"""

import subprocess
import re
import pytest
import numpy as np

from ModelInfrenceFramework import TestOcrDetInference
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
 sed -i '/config.enable_tensorrt_engine/i\\            config.collect_shape_range_info("shape_range.pbtxt")' \
 tools/infer/utility.py; \
 sed -i '/min_subgraph_size=args.min_subgraph_size)/a\\
            config.enable_tuned_tensorrt_dynamic_shape("shape_range.pbtxt", True)' \
 tools/infer/utility.py"""
    )


def setup_function():
    """
    clean process
    """
    clean_process()


def test_ch_PP_OCRv2_det_infer():
    """
    test case
    """
    model = TestOcrDetInference(model="ch_PP-OCRv2_det", infer_imgs="./doc/imgs_en/img_10.jpg")
    model.test_get_ocr_det_inference_model(category="PP-OCRv2/chinese")
    model.test_ocr_det_predict(
        expect_det_bbox=[
            [[41, 85], [201, 74], [203, 101], [43, 113]],
            [[37, 55], [202, 49], [203, 76], [38, 82]],
            [[18, 14], [241, 26], [239, 57], [17, 46]],
        ]
    )


def test_ch_PP_OCRv2_det_slim_quant_infer():
    """
    test case
    """
    model = TestOcrDetInference(model="ch_PP-OCRv2_det_slim_quant", infer_imgs="./doc/imgs_en/img_10.jpg")
    model.test_get_ocr_det_inference_model(category="PP-OCRv2/chinese")
    model.test_ocr_det_predict(
        expect_det_bbox=[
            [[39, 88], [204, 76], [206, 100], [41, 112]],
            [[37, 57], [202, 53], [203, 75], [38, 80]],
            [[35, 22], [219, 27], [218, 49], [34, 45]],
        ]
    )


def test_ch_ppocr_mobile_v2_det_infer():
    """
    test case
    """
    model = TestOcrDetInference(model="ch_ppocr_mobile_v2.0_det", infer_imgs="./doc/imgs_en/img_10.jpg")
    model.test_get_ocr_det_inference_model(category="dygraph_v2.0/ch")
    model.test_ocr_det_predict(
        expect_det_bbox=[[[45, 86], [196, 78], [198, 100], [46, 109]], [[40, 54], [201, 53], [201, 77], [40, 78]]]
    )


def test_ch_ppocr_mobile_v2_det_prune_infer():
    """
    test case
    """
    model = TestOcrDetInference(model="ch_ppocr_mobile_v2.0_det_prune", infer_imgs="./doc/imgs_en/img_10.jpg")
    model.test_get_ocr_det_inference_model(category="dygraph_v2.0/slim")
    model.test_ocr_det_predict(
        expect_det_bbox=[
            [[46, 90], [197, 79], [198, 96], [47, 108]],
            [[41, 57], [200, 57], [200, 74], [41, 74]],
            [[39, 29], [203, 29], [203, 42], [39, 42]],
        ]
    )


def test_ch_ppocr_server_v2_det_infer():
    """
    test case
    """
    model = TestOcrDetInference(model="ch_ppocr_server_v2.0_det", infer_imgs="./doc/imgs_en/img_10.jpg")
    model.test_get_ocr_det_inference_model(category="dygraph_v2.0/ch")
    model.test_ocr_det_predict(
        expect_det_bbox=[
            [[41, 88], [198, 78], [199, 100], [42, 110]],
            [[40, 54], [203, 53], [203, 75], [40, 77]],
            [[30, 21], [223, 27], [222, 49], [29, 43]],
        ]
    )


def test_det_r50_vd_east_v2_infer():
    """
    test case
    """
    model = TestOcrDetInference(
        model="det_r50_vd_east_v2.0",
        infer_imgs="./doc/imgs_en/img_10.jpg",
        yaml="configs/det/det_r50_vd_east.yml",
        algorithm="EAST",
    )
    model.test_get_ocr_det_train_model(category="dygraph_v2.0/en")
    model.test_ocr_det_export_model()
    model.test_ocr_det_predict(
        expect_det_bbox=[
            [[44, 88], [145, 78], [149, 104], [47, 113]],
            [[32, 20], [105, 21], [107, 48], [34, 46]],
            [[151, 82], [199, 79], [201, 99], [152, 101]],
            [[37, 57], [95, 54], [98, 77], [39, 80]],
            [[200, 29], [250, 29], [252, 47], [202, 46]],
            [[110, 25], [157, 24], [159, 47], [112, 47]],
            [[162, 27], [196, 26], [198, 47], [163, 48]],
            [[101, 55], [140, 54], [141, 76], [102, 77]],
            [[143, 55], [183, 52], [185, 73], [144, 75]],
            [[184, 50], [205, 50], [206, 75], [185, 73]],
        ]
    )


def test_det_r50_vd_sast_icdar15_v2_infer():
    """
    test case
    """
    model = TestOcrDetInference(
        model="det_r50_vd_sast_icdar15_v2.0",
        infer_imgs="./doc/imgs_en/img_10.jpg",
        yaml="configs/det/det_r50_vd_sast_icdar15.yml",
        algorithm="SAST",
    )
    model.test_get_ocr_det_train_model(category="dygraph_v2.0/en")
    model.test_ocr_det_export_model()
    model.test_ocr_det_predict(
        expect_det_bbox=[
            [[36, 88], [151, 79], [153, 103], [38, 112]],
            [[95, 55], [212, 51], [213, 75], [97, 82]],
            [[148, 82], [205, 79], [206, 99], [149, 102]],
            [[31, 54], [102, 52], [103, 78], [32, 80]],
            [[25, 19], [106, 23], [108, 48], [28, 46]],
            [[106, 24], [161, 23], [162, 47], [108, 47]],
            [[197, 26], [256, 27], [257, 48], [199, 49]],
            [[156, 27], [201, 26], [202, 47], [159, 51]],
        ]
    )


def test_det_r50_vd_sast_totaltext_v2_infer():
    """
    test case
    """
    model = TestOcrDetInference(
        model="det_r50_vd_sast_totaltext_v2.0",
        infer_imgs="./doc/imgs_en/img_10.jpg",
        yaml="configs/det/det_r50_vd_sast_totaltext.yml",
        algorithm="SAST",
    )
    model.test_get_ocr_det_train_model(category="dygraph_v2.0/en")
    model.test_ocr_det_export_model()
    model.test_ocr_det_predict(
        expect_det_bbox=[
            [[35, 91], [150, 77], [153, 100], [40, 113]],
            [[25, 18], [107, 19], [110, 45], [27, 47]],
            [[34, 55], [101, 52], [104, 77], [38, 81]],
            [[147, 82], [197, 79], [207, 96], [148, 101]],
            [[198, 27], [255, 28], [257, 48], [198, 47]],
            [[107, 22], [161, 26], [161, 47], [109, 45]],
            [[137, 53], [186, 51], [188, 72], [138, 76]],
            [[157, 27], [201, 26], [201, 47], [159, 48]],
            [[182, 52], [211, 48], [213, 72], [183, 75]],
            [[96, 53], [142, 51], [142, 75], [98, 77]],
        ]
    )


def test_det_r50_vd_pse_v2_infer():
    """
    test case
    """
    model = TestOcrDetInference(
        model="det_r50_vd_pse_v2.0",
        infer_imgs="./doc/imgs_en/img_10.jpg",
        yaml="configs/det/det_r50_vd_pse.yml",
        algorithm="PSE",
    )
    model.test_get_ocr_det_train_model(category="dygraph_v2.1/en_det")
    model.test_ocr_det_export_model()
    model.test_ocr_det_predict(
        expect_det_bbox=[
            [[30, 17], [109, 24], [107, 51], [28, 44]],
            [[110, 23], [162, 26], [160, 49], [108, 46]],
            [[160, 26], [199, 26], [199, 50], [160, 50]],
            [[199, 28], [251, 28], [251, 46], [199, 46]],
            [[36, 52], [100, 52], [100, 78], [36, 78]],
            [[143, 50], [184, 50], [184, 75], [143, 75]],
            [[184, 52], [205, 52], [205, 77], [184, 77]],
            [[100, 52], [141, 52], [141, 78], [100, 78]],
            [[151, 75], [201, 75], [201, 101], [151, 101]],
            [[41, 80], [149, 74], [150, 104], [43, 110]],
        ]
    )
