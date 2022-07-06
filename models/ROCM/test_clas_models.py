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

from RocmTestFramework import TestClassModel
from RocmTestFramework import RepoInit
from RocmTestFramework import RepoRemove
from RocmTestFramework import RepoDataset
from RocmTestFramework import clean_process


def setup_module():
    """
    function
    """
    RepoInit(repo="PaddleClas")
    RepoDataset(
        cmd="rm -rf /root/.visualdl/conf;cd PaddleClas; \
            python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple; \
            cd dataset; rm -rf ILSVRC2012; ln -s /data/ILSVRC2012 ILSVRC2012; rm -rf flowers102; \
            ln -s /data/flowers102 flowers102; cd .."
    )


def teardown_module():
    """
    function
    """
    RepoRemove(repo="PaddleClas")


def setup_function():
    """
    function
    """
    clean_process()


def test_ResNet50():
    """
    ResNet50 test case
    """
    model = TestClassModel(model="ResNet50", yaml="ppcls/configs/ImageNet/ResNet/ResNet50.yaml")
    model.test_class_train()
    model.test_get_pretrained_model()
    model.test_class_export_model()
    model.test_class_predict(expect_id="153, 332, 229, 204, 265", expect_score="0.41, 0.39, 0.05, 0.04, 0.04")


def test_ResNet101():
    """
    ResNet101 test case
    """
    model = TestClassModel(model="ResNet101", yaml="ppcls/configs/ImageNet/ResNet/ResNet101.yaml")
    model.test_class_train()
    model.test_get_pretrained_model()
    model.test_class_export_model()
    model.test_class_predict(expect_id="153, 332, 229, 204, 265", expect_score="0.73, 0.13, 0.05, 0.03, 0.02")


def test_AlexNet():
    """
    AlexNet test case
    """
    model = TestClassModel(model="AlexNet", yaml="ppcls/configs/ImageNet/AlexNet/AlexNet.yaml")
    model.test_class_train()
    model.test_get_pretrained_model()
    model.test_class_export_model()
    model.test_class_predict(expect_id="153, 204, 265, 283, 154", expect_score="0.33, 0.14, 0.14, 0.07, 0.07")


def test_MobileNetV3_large_x1_0():
    """
    MobileNetV3_large_x1_0 test case
    """
    model = TestClassModel(
        model="MobileNetV3_large_x1_0", yaml="ppcls/configs/ImageNet/MobileNetV3/MobileNetV3_large_x1_0.yaml"
    )
    model.test_get_pretrained_model()
    model.test_class_export_model()
    model.test_class_predict(expect_id="153, 283, 332, 204, 229", expect_score="0.55, 0.09, 0.05, 0.04, 0.01")


def test_GoogLeNet():
    """
    GoogLeNet test case
    """
    model = TestClassModel(model="GoogLeNet", yaml="ppcls/configs/ImageNet/Inception/GoogLeNet.yaml")
    model.test_class_train()
    model.test_get_pretrained_model()
    model.test_class_export_model()
    model.test_class_predict(expect_id="332, 283, 153, 204, 338", expect_score="0.49, 0.28, 0.09, 0.06, 0.03")


def test_InceptionV4():
    """
    InceptionV4 test case
    """
    model = TestClassModel(model="InceptionV4", yaml="ppcls/configs/ImageNet/Inception/InceptionV4.yaml")
    model.test_class_train()
    model.test_get_pretrained_model()
    model.test_class_export_model()
    model.test_class_predict(expect_id="153, 332, 265, 204, 229", expect_score="0.94, 0.06, 0.00, 0.00, 0.00")


def test_VGG16():
    """
    VGG16 test case
    """
    model = TestClassModel(model="VGG16", yaml="ppcls/configs/ImageNet/VGG/VGG16.yaml")
    model.test_class_train()
    model.test_get_pretrained_model()
    model.test_class_export_model()
    model.test_class_predict(expect_id="265, 153, 903, 332, 204", expect_score="0.26, 0.25, 0.15, 0.12, 0.09")


def test_SE_ResNet50_vd():
    """
    SE_ResNet50_vd test case
    """
    model = TestClassModel(model="SE_ResNet50_vd", yaml="ppcls/configs/ImageNet/SENet/SE_ResNet50_vd.yaml")
    model.test_class_train()
    model.test_get_pretrained_model()
    model.test_class_export_model()
    model.test_class_predict(expect_id="332, 153, 283, 204, 338", expect_score="0.57, 0.12, 0.01, 0.01, 0.01")


def test_DenseNet121():
    """
    DenseNet121 test case
    """
    model = TestClassModel(model="DenseNet121", yaml="ppcls/configs/ImageNet/DenseNet/DenseNet121.yaml")
    model.test_class_train()
    model.test_get_pretrained_model()
    model.test_class_export_model()
    model.test_class_predict(expect_id="153, 265, 229, 332, 204", expect_score="0.86, 0.08, 0.03, 0.01, 0.01")
