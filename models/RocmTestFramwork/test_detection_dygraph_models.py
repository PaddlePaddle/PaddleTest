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
import numpy as np
import pytest

from RocmTestFramework import TestDetectionDygraphModel
from RocmTestFramework import RepoInit
from RocmTestFramework import RepoRemove
from RocmTestFramework import RepoDataset
from RocmTestFramework import clean_process


def setup_module():
    """
    setup
    """
    RepoInit(repo="PaddleDetection")
    RepoDataset(
        cmd="""cd PaddleDetection/dataset;
                     rm -rf wider_face voc coco;
                     ln -s /data/VOC_Paddle voc;
                     ln -s /data/COCO17 coco;
                     ln -s /data/wider_face wider_face;
                     cd ..;
                     python -m pip install -U numpy==1.20.3 -i https://pypi.tuna.tsinghua.edu.cn/simple;
                     sed -i "1i find_unused_parameters: True" configs/face_detection/_base_/blazeface.yml;
                     sed -ie '/records = records\\[:10\\]/d'  ppdet/data/source/coco.py;
                     sed -ie '/records = records\\[:10\\]/d'  ppdet/data/source/voc.py;
                     sed -ie '/records = records\\[:10\\]/d'  ppdet/data/source/widerface.py;
                     sed -i '/samples in file/i\\        records = records[:10]'  ppdet/data/source/coco.py;
                     sed -i '/samples in file/i\\        records = records[:10]'  ppdet/data/source/voc.py;
                     sed -i '/samples in file/i\\        records = records[:10]'  ppdet/data/source/widerface.py;"""
    )


def teardown_module():
    """
    teardown
    """
    RepoRemove(repo="PaddleDetection")


def setup_function():
    """
    setup
    """
    clean_process()


def test_faster_rcnn_r50_1x_coco():
    """
    faster_rcnn_r50_1x_coco test case
    """
    model = TestDetectionDygraphModel(
        model="faster_rcnn_r50_1x_coco", yaml="configs/faster_rcnn/faster_rcnn_r50_1x_coco.yml"
    )
    model.test_detection_train()


def test_faster_rcnn_r50_fpn_1x_coco():
    """
    faster_rcnn_r50_fpn_1x_coco test case
    """
    model = TestDetectionDygraphModel(
        model="faster_rcnn_r50_fpn_1x_coco", yaml="configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.yml"
    )
    model.test_detection_train()


def test_mask_rcnn_r50_1x_coco():
    """
    mask_rcnn_r50_1x_coco test case
    """
    model = TestDetectionDygraphModel(model="mask_rcnn_r50_1x_coco", yaml="configs/mask_rcnn/mask_rcnn_r50_1x_coco.yml")
    model.test_detection_train()


def test_mask_rcnn_r50_fpn_1x_coco():
    """
    mask_rcnn_r50_fpn_1x_coco test case
    """
    model = TestDetectionDygraphModel(
        model="mask_rcnn_r50_fpn_1x_coco", yaml="configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.yml"
    )
    model.test_detection_train()


def test_cascade_rcnn_r50_fpn_1x_coco():
    """
    cascade_rcnn_r50_fpn_1x_coco test case
    """
    model = TestDetectionDygraphModel(
        model="cascade_rcnn_r50_fpn_1x_coco", yaml="configs/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco.yml"
    )
    model.test_detection_train()


def test_cascade_rcnn_r50_fpn_1x_coco():
    """
    cascade_rcnn_r50_fpn_1x_coco test case
    """
    model = TestDetectionDygraphModel(
        model="cascade_rcnn_r50_fpn_1x_coco", yaml="configs/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco.yml"
    )
    model.test_detection_train()


def test_ssd_vgg16_300_240e_voc():
    """
    ssd_vgg16_300_240e_voc test case
    """
    model = TestDetectionDygraphModel(model="ssd_vgg16_300_240e_voc", yaml="configs/ssd/ssd_vgg16_300_240e_voc.yml")
    model.test_detection_train()


def test_yolov3_darknet53_270e_coco():
    """
    yolov3_darknet53_270e_coco test case
    """
    model = TestDetectionDygraphModel(
        model="yolov3_darknet53_270e_coco", yaml="configs/yolov3/yolov3_darknet53_270e_coco.yml"
    )
    model.test_detection_train()


def test_blazeface_1000e():
    """
    blazeface_1000e test case
    """
    model = TestDetectionDygraphModel(model="blazeface_1000e", yaml="configs/face_detection/blazeface_1000e.yml")
    model.test_detection_train()


def test_fcos_dcn_r50_fpn_1x_coco():
    """
    fcos_dcn_r50_fpn_1x_coco test case
    """
    model = TestDetectionDygraphModel(
        model="fcos_dcn_r50_fpn_1x_coco", yaml="configs/fcos/fcos_dcn_r50_fpn_1x_coco.yml"
    )
    model.test_detection_train()


def test_ttfnet_darknet53_1x_coco():
    """
    ttfnet_darknet53_1x_coco test case
    """
    model = TestDetectionDygraphModel(
        model="ttfnet_darknet53_1x_coco", yaml="configs/ttfnet/ttfnet_darknet53_1x_coco.yml"
    )
    model.test_detection_train()


def test_ppyolo_r50vd_dcn_voc():
    """
    ppyolo_r50vd_dcn_voc test case
    """
    model = TestDetectionDygraphModel(model="ppyolo_r50vd_dcn_voc", yaml="configs/ppyolo/ppyolo_r50vd_dcn_voc.yml")
    model.test_detection_train()


def test_solov2_r50_fpn_1x_coco():
    """
    solov2_r50_fpn_1x_coco test case
    """
    model = TestDetectionDygraphModel(model="solov2_r50_fpn_1x_coco", yaml="configs/solov2/solov2_r50_fpn_1x_coco.yml")
    model.test_detection_train()
