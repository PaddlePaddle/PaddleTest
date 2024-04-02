#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
print
"""
import os
import paddle
import paddle.nn as nn
from paddle import ParamAttr
from paddle.nn import AdaptiveAvgPool2D, BatchNorm, Conv2D, Dropout, Linear
from paddle.regularizer import L2Decay
from paddle.nn.initializer import KaimingNormal
from ppcls.arch.backbone.base.theseus_layer import TheseusLayer
import ppcls.arch.backbone.legendary_models.esnet as esnet
import ppcls.arch.backbone.legendary_models.hrnet as hrnet
import ppcls.arch.backbone.legendary_models.inception_v3 as inception_v3
import ppcls.arch.backbone.legendary_models.mobilenet_v1 as mobilenet_v1
import ppcls.arch.backbone.legendary_models.mobilenet_v3 as mobilenet_v3
import ppcls.arch.backbone.legendary_models.pp_lcnet as pp_lcnet
import ppcls.arch.backbone.legendary_models.resnet as resnet
import ppcls.arch.backbone.legendary_models.vgg as vgg


model_zoo = {
    "ESNet_x0_25": esnet.ESNet_x0_25,
    "ESNet_x0_5": esnet.ESNet_x0_5,
    "ESNet_x0_75": esnet.ESNet_x0_75,
    "ESNet_x1_0": esnet.ESNet_x1_0,
    "HRNet_W18_C": hrnet.HRNet_W18_C,
    "HRNet_W30_C": hrnet.HRNet_W30_C,
    "HRNet_W32_C": hrnet.HRNet_W32_C,
    "HRNet_W40_C": hrnet.HRNet_W40_C,
    "HRNet_W44_C": hrnet.HRNet_W44_C,
    "HRNet_W48_C": hrnet.HRNet_W48_C,
    "HRNet_W64_C": hrnet.HRNet_W64_C,
    "InceptionV3": inception_v3.InceptionV3,
    "MobileNetV1_x0_25": mobilenet_v1.MobileNetV1_x0_25,
    "MobileNetV1_x0_5": mobilenet_v1.MobileNetV1_x0_5,
    "MobileNetV1_x0_75": mobilenet_v1.MobileNetV1_x0_75,
    "MobileNetV1": mobilenet_v1.MobileNetV1,
    "MobileNetV3_small_x0_35": mobilenet_v3.MobileNetV3_small_x0_35,
    "MobileNetV3_small_x0_5": mobilenet_v3.MobileNetV3_small_x0_5,
    "MobileNetV3_small_x0_75": mobilenet_v3.MobileNetV3_small_x0_75,
    "MobileNetV3_small_x1_0": mobilenet_v3.MobileNetV3_small_x1_0,
    "MobileNetV3_small_x1_25": mobilenet_v3.MobileNetV3_small_x1_25,
    "MobileNetV3_large_x0_35": mobilenet_v3.MobileNetV3_large_x0_35,
    "MobileNetV3_large_x0_5": mobilenet_v3.MobileNetV3_large_x0_5,
    "MobileNetV3_large_x0_75": mobilenet_v3.MobileNetV3_large_x0_75,
    "MobileNetV3_large_x1_0": mobilenet_v3.MobileNetV3_large_x1_0,
    "MobileNetV3_large_x1_25": mobilenet_v3.MobileNetV3_large_x1_25,
    "PPLCNet_x0_25": pp_lcnet.PPLCNet_x0_25,
    "PPLCNet_x0_35": pp_lcnet.PPLCNet_x0_35,
    "PPLCNet_x0_5": pp_lcnet.PPLCNet_x0_5,
    "PPLCNet_x0_75": pp_lcnet.PPLCNet_x0_75,
    "PPLCNet_x1_0": pp_lcnet.PPLCNet_x1_0,
    "PPLCNet_x1_5": pp_lcnet.PPLCNet_x1_5,
    "PPLCNet_x2_0": pp_lcnet.PPLCNet_x2_0,
    "PPLCNet_x2_5": pp_lcnet.PPLCNet_x2_5,
    "ResNet18": resnet.ResNet18,
    "ResNet18_vd": resnet.ResNet18_vd,
    "ResNet34": resnet.ResNet34,
    "ResNet34_vd": resnet.ResNet34_vd,
    "ResNet50": resnet.ResNet50,
    "ResNet50_vd": resnet.ResNet50_vd,
    "ResNet101": resnet.ResNet101,
    "ResNet101_vd": resnet.ResNet101_vd,
    "ResNet152": resnet.ResNet152,
    "ResNet152_vd": resnet.ResNet152_vd,
    "ResNet200_vd": resnet.ResNet200_vd,
    "VGG11": vgg.VGG11,
    "VGG13": vgg.VGG13,
    "VGG16": vgg.VGG16,
    "VGG19": vgg.VGG19,
}


# 'HRNet_W60_C': hrnet.HRNet_W60_C, 'SE_HRNet_W18_C': hrnet.SE_HRNet_W18_C, 'SE_HRNet_W30_C': hrnet.SE_HRNet_W30_C,
# 'SE_HRNet_W32_C': hrnet.SE_HRNet_W32_C, 'SE_HRNet_W40_C': hrnet.SE_HRNet_W40_C,
# 'SE_HRNet_W44_C': hrnet.SE_HRNet_W44_C, 'SE_HRNet_W48_C': hrnet.SE_HRNet_W48_C,
# 'SE_HRNet_W60_C': hrnet.SE_HRNet_W60_C,
# 'SE_HRNet_W64_C': hrnet.SE_HRNet_W64_C,

for k, v in model_zoo.items():
    print("{} start".format(k))
    model = v()
    doc = open(os.path.join("model_structure", k), "w")
    print(model, file=doc)
    print("{} end".format(k))
