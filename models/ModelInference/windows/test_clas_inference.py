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

from ModelInfrenceFramework import TestClasInference
from ModelInfrenceFramework import RepoInit
from ModelInfrenceFramework import clean_process


def setup_module():
    """
    git clone repo and install dependency
    """
    RepoInit(repo="PaddleClas", branch="develop")


def test_ResNet50_vd():
    """
    ResNet50_vd test case
    """
    model = TestClasInference(model="ResNet50_vd", yaml="ppcls/configs/ImageNet/ResNet/ResNet50_vd.yaml")
    model.test_get_pretrained_model()
    model.test_clas_export_model()
    model.test_clas_predict(expect_id="153, 204, 229, 332, 155", expect_score="0.69, 0.10, 0.02, 0.01, 0.01")


def test_ResNet50():
    """
    ResNet50 test case
    """
    model = TestClasInference(model="ResNet50", yaml="ppcls/configs/ImageNet/ResNet/ResNet50.yaml")
    model.test_get_pretrained_model()
    model.test_clas_export_model()
    model.test_clas_predict(expect_id="153, 332, 229, 204, 265", expect_score="0.41, 0.39, 0.05, 0.04, 0.04")


def test_AlexNet():
    """
    AlexNet test case
    """
    model = TestClasInference(model="AlexNet", yaml="ppcls/configs/ImageNet/AlexNet/AlexNet.yaml")
    model.test_get_pretrained_model()
    model.test_clas_export_model()
    model.test_clas_predict(expect_id="153, 204, 265, 283, 154", expect_score="0.33, 0.14, 0.14, 0.07, 0.07")


def test_MobileNetV3_large_x1_0():
    """
    MobileNetV3_large_x1_0 test case
    """
    model = TestClasInference(
        model="MobileNetV3_large_x1_0", yaml="ppcls/configs/ImageNet/MobileNetV3/MobileNetV3_large_x1_0.yaml"
    )
    model.test_get_pretrained_model()
    model.test_clas_export_model()
    model.test_clas_predict(expect_id="153, 283, 332, 204, 229", expect_score="0.55, 0.09, 0.05, 0.04, 0.01")


def test_GoogLeNet():
    """
    GoogLeNet test case
    """
    model = TestClasInference(model="GoogLeNet", yaml="ppcls/configs/ImageNet/Inception/GoogLeNet.yaml")
    model.test_get_pretrained_model()
    model.test_clas_export_model()
    model.test_clas_predict(expect_id="332, 283, 153, 204, 338", expect_score="0.49, 0.28, 0.09, 0.06, 0.03")


def test_VGG16():
    """
    VGG16 test case
    """
    model = TestClasInference(model="VGG16", yaml="ppcls/configs/ImageNet/VGG/VGG16.yaml")
    model.test_get_pretrained_model()
    model.test_clas_export_model()
    model.test_clas_predict(expect_id="265, 153, 903, 332, 204", expect_score="0.26, 0.25, 0.15, 0.12, 0.09")


def test_SE_ResNet50_vd():
    """
    SE_ResNet50_vd test case
    """
    model = TestClasInference(model="SE_ResNet50_vd", yaml="ppcls/configs/ImageNet/SENet/SE_ResNet50_vd.yaml")
    model.test_get_pretrained_model()
    model.test_clas_export_model()
    model.test_clas_predict(expect_id="332, 153, 283, 204, 338", expect_score="0.57, 0.12, 0.01, 0.01, 0.01")


def test_DenseNet121():
    """
    DenseNet121 test case
    """
    model = TestClasInference(model="DenseNet121", yaml="ppcls/configs/ImageNet/DenseNet/DenseNet121.yaml")
    model.test_get_pretrained_model()
    model.test_clas_export_model()
    model.test_clas_predict(expect_id="153, 265, 229, 332, 204", expect_score="0.86, 0.08, 0.03, 0.01, 0.01")


def test_DarkNet53():
    """
    DarkNet53 test case
    """
    model = TestClasInference(model="DarkNet53", yaml="ppcls/configs/ImageNet/DarkNet/DarkNet53.yaml")
    model.test_get_pretrained_model()
    model.test_clas_export_model()
    model.test_clas_predict(expect_id="153, 332, 283, 265, 338", expect_score="0.46, 0.24, 0.04, 0.03, 0.02")


def test_DeiT_base_distilled_patch16_224():
    """
    DeiT_base_distilled_patch16_224 test case
    """
    model = TestClasInference(
        model="DeiT_base_distilled_patch16_224", yaml="ppcls/configs/ImageNet/DeiT/DeiT_base_distilled_patch16_224.yaml"
    )
    model.test_get_pretrained_model()
    model.test_clas_export_model()
    model.test_clas_predict(expect_id="283, 153, 332, 204, 265", expect_score="0.72, 0.16, 0.05, 0.02, 0.02")


def test_DPN68():
    """
    DPN68 test case
    """
    model = TestClasInference(model="DPN68", yaml="ppcls/configs/ImageNet/DPN/DPN68.yaml")
    model.test_get_pretrained_model()
    model.test_clas_export_model()
    model.test_clas_predict(expect_id="332, 153, 204, 338, 155", expect_score="0.29, 0.09, 0.08, 0.02, 0.02")


def test_EfficientNetB0():
    """
    EfficientNetB0 test case
    """
    model = TestClasInference(model="EfficientNetB0", yaml="ppcls/configs/ImageNet/EfficientNet/EfficientNetB0.yaml")
    model.test_get_pretrained_model()
    model.test_clas_export_model()
    model.test_clas_predict(expect_id="153, 332, 204, 155, 283", expect_score="0.41, 0.12, 0.10, 0.05, 0.02")


def test_GhostNet_x1_0():
    """
    GhostNet_x1_0 test case
    """
    model = TestClasInference(model="GhostNet_x1_0", yaml="ppcls/configs/ImageNet/GhostNet/GhostNet_x1_0.yaml")
    model.test_get_pretrained_model()
    model.test_clas_export_model()
    model.test_clas_predict(expect_id="153, 204, 332, 283, 265", expect_score="0.19, 0.10, 0.07, 0.05, 0.02")


def test_LeViT_128():
    """
    LeViT_128 test case
    """
    model = TestClasInference(model="LeViT_128", yaml="ppcls/configs/ImageNet/LeViT/LeViT_128.yaml")
    model.test_get_pretrained_model()
    model.test_clas_export_model()
    model.test_clas_predict(expect_id="153, 204, 332, 283, 229", expect_score="0.24, 0.07, 0.07, 0.04, 0.04")


def test_pcpvt_base():
    """
    pcpvt_base test case
    """
    model = TestClasInference(model="pcpvt_base", yaml="ppcls/configs/ImageNet/Twins/pcpvt_base.yaml")
    model.test_get_pretrained_model()
    model.test_clas_export_model()
    model.test_clas_predict(expect_id="153, 283, 204, 332, 154", expect_score="0.42, 0.17, 0.08, 0.04, 0.01")


def test_PPLCNet_x1_0():
    """
    PPLCNet_x1_0 test case
    """
    model = TestClasInference(model="PPLCNet_x1_0", yaml="ppcls/configs/ImageNet/PPLCNet/PPLCNet_x1_0.yaml")
    model.test_get_pretrained_model()
    model.test_clas_export_model()
    model.test_clas_predict(expect_id="153, 265, 204, 283, 229", expect_score="0.61, 0.11, 0.05, 0.03, 0.02")


def test_ResNeXt101_32x16d_wsl():
    """
    ResNeXt101_32x16d_wsl test case
    """
    model = TestClasInference(
        model="ResNeXt101_32x16d_wsl", yaml="ppcls/configs/ImageNet/ResNeXt101_wsl/ResNeXt101_32x16d_wsl.yaml"
    )
    model.test_get_pretrained_model()
    model.test_clas_export_model()
    model.test_clas_predict(expect_id="332, 153, 265, 229, 203", expect_score="1.00, 0.00, 0.00, 0.00, 0.00")


def test_ShuffleNetV2_x1_0():
    """
    ShuffleNetV2 test case
    """
    model = TestClasInference(
        model="ShuffleNetV2_x1_0", yaml="ppcls/configs/ImageNet/ShuffleNet/ShuffleNetV2_x1_0.yaml"
    )
    model.test_get_pretrained_model()
    model.test_clas_export_model()
    model.test_clas_predict(expect_id="153, 204, 332, 265, 283", expect_score="0.74, 0.09, 0.06, 0.03, 0.02")


def test_SqueezeNet1_0():
    """
    SqueezeNet1_0 test case
    """
    model = TestClasInference(model="SqueezeNet1_0", yaml="ppcls/configs/ImageNet/SqueezeNet/SqueezeNet1_0.yaml")
    model.test_get_pretrained_model()
    model.test_clas_export_model()
    model.test_clas_predict(expect_id="283, 153, 903, 154, 259", expect_score="0.21, 0.20, 0.11, 0.10, 0.07")


def test_SwinTransformer_base_patch4_window7_224():
    """
    SwinTransformer_base_patch4_window7_224 test case
    """
    model = TestClasInference(
        model="SwinTransformer_base_patch4_window7_224",
        yaml="ppcls/configs/ImageNet/SwinTransformer/SwinTransformer_base_patch4_window7_224.yaml",
    )
    model.test_get_pretrained_model()
    model.test_clas_export_model()
    model.test_clas_predict(expect_id="332, 153, 283, 154, 204", expect_score="0.76, 0.02, 0.01, 0.00, 0.00")


def test_TNT_small():
    """
    TNT_small test case
    """
    model = TestClasInference(model="TNT_small", yaml="ppcls/configs/ImageNet/TNT/TNT_small.yaml")
    model.test_get_pretrained_model()
    model.test_clas_export_model()
    model.test_clas_predict(expect_id="332, 153, 190, 196, 283", expect_score="0.24, 0.22, 0.17, 0.06, 0.04")


def test_alt_gvt_base():
    """
    alt_gvt_base test case
    """
    model = TestClasInference(model="alt_gvt_base", yaml="ppcls/configs/ImageNet/Twins/alt_gvt_base.yaml")
    model.test_get_pretrained_model()
    model.test_clas_export_model()
    model.test_clas_predict(expect_id="283, 332, 153, 265, 154", expect_score="0.48, 0.21, 0.10, 0.01, 0.01")


def test_ViT_base_patch16_224():
    """
    ViT_base_patch16_224 test case
    """
    model = TestClasInference(
        model="ViT_base_patch16_224", yaml="ppcls/configs/ImageNet/VisionTransformer/ViT_base_patch16_224.yaml"
    )
    model.test_get_pretrained_model()
    model.test_clas_export_model()
    model.test_clas_predict(expect_id="850, 332, 153, 265, 204", expect_score="0.51, 0.11, 0.06, 0.04, 0.03")


@pytest.mark.skip(reson="InvalidArgumentError: Input shapes are inconsistent with the model.")
def test_Xception41_deeplab():
    """
    Xception41 test case
    """
    model = TestClasInference(
        model="Xception41_deeplab", yaml="ppcls/configs/ImageNet/Xception/Xception41_deeplab.yaml"
    )
    model.test_get_pretrained_model()
    model.test_clas_export_model()
    model.test_clas_predict(expect_id="332, 153, 283, 265, 204", expect_score="0.81, 0.17, 0.00, 0.00, 0.00")
