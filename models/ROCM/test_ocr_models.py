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

import re
import subprocess
import pytest
import numpy as np

from RocmTestFramework import TestOcrModel
from RocmTestFramework import RepoInit
from RocmTestFramework import RepoRemove
from RocmTestFramework import RepoDataset
from RocmTestFramework import clean_process


def setup_module():
    """
    function
    """
    RepoInit(repo="PaddleOCR")
    RepoDataset(
        cmd="""cd PaddleOCR; \
                python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple; \
                ln -s /data train_data;
                wget -q -P ./pretrain_models/ \
            https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV3_large_x0_5_pretrained.pdparams; \
                wget -q https://paddleocr.bj.bcebos.com/dygraph_v2.0/pgnet/en_server_pgnetA.tar; \
                tar xf en_server_pgnetA.tar; mv en_server_pgnetA e2e_r50_vd_pg; \
                mkdir output; mv e2e_r50_vd_pg output; \
                mv output/e2e_r50_vd_pg/best_accuracy.pdparams output/e2e_r50_vd_pg/latest.pdparams; \
                mv output/e2e_r50_vd_pg/best_accuracy.pdopt output/e2e_r50_vd_pg/latest.pdopt"""
    )


def teardown_module():
    """
    function
    """
    RepoRemove(repo="PaddleOCR")


def setup_function():
    """
    function
    """
    clean_process()


def test_rec_mv3_none_bilstm_ctc():
    """
    rec_mv3_none_bilstm_ctc test case
    """
    model = TestOcrModel(model="rec_mv3_none_bilstm_ctc", yaml="configs/rec/rec_mv3_none_bilstm_ctc.yml")
    model.test_ocr_train()
    model.test_ocr_eval()
    model.test_ocr_rec_infer()
    model.test_ocr_export_model()
    model.test_ocr_rec_predict()


def test_det_mv3_db():
    """
    det_mv3_db test case
    """
    model = TestOcrModel(model="det_mv3_db", yaml="configs/det/det_mv3_db.yml")
    model.test_ocr_train()
    model.test_ocr_eval()
    model.test_ocr_det_infer()
    model.test_ocr_export_model()
    model.test_ocr_det_predict()


def test_e2e_r50_vd_pg():
    """
    e2e_r50_vd_pg test case
    """
    model = TestOcrModel(model="e2e_r50_vd_pg", yaml="configs/e2e/e2e_r50_vd_pg.yml")
    model.test_ocr_eval()
    model.test_ocr_e2e_infer()
    model.test_ocr_export_model()
    model.test_ocr_e2e_predict()


def test_cls_mv3():
    """
    cls_mv3 test case
    """
    model = TestOcrModel(model="cls_mv3", yaml="configs/cls/cls_mv3.yml")
    model.test_ocr_train()
    model.test_ocr_eval()
    model.test_ocr_cls_infer()
    model.test_ocr_export_model()
    model.test_ocr_cls_predict()
