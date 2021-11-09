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

import pytest
import numpy as np
import subprocess
import re

from RocmTestFramework import TestDetectionStaticModel
from RocmTestFramework import RepoInit
from RocmTestFramework import RepoRemove
from RocmTestFramework import RepoDataset
from RocmTestFramework import clean_process

def setup_module():
    """
    """
    RepoInit(repo='PaddleDetection')
    RepoDataset(cmd='''cd PaddleDetection/static/dataset;  
                     rm -rf wider_face voc coco; 
                     ln -s /data/VOC_Paddle voc; 
                     ln -s /data/COCO17 coco; 
                     ln -s /data/wider_face wider_face; 
                     cd ..;
                     sed -ie '/records = records\[:10\]/d'  ppdet/data/source/coco.py;
                     sed -ie '/records = records\[:10\]/d'  ppdet/data/source/voc.py;
                     sed -ie '/records = records\[:10\]/d'  ppdet/data/source/widerface.py;
                     sed -i '/samples in file/i\        records = records[:10]'  ppdet/data/source/coco.py;
                     sed -i '/samples in file/i\        records = records[:10]'  ppdet/data/source/voc.py;
                     sed -i '/samples in file/i\        records = records[:10]'  ppdet/data/source/widerface.py;''')

def teardown_module():
    """
    """
    RepoRemove(repo='PaddleDetection')

def setup_function():
    clean_process()    

def test_libra_rcnn_r50_vd_fpn_1x():
    """
    libra_rcnn_r50_vd_fpn_1x test case
    """
    model = TestDetectionStaticModel(model='libra_rcnn_r50_vd_fpn_1x', yaml='configs/libra_rcnn/libra_rcnn_r50_vd_fpn_1x.yml')
    model.test_detection_train()

def test_retinanet_r50_fpn_1x():
    """
    retinanet_r50_fpn_1x test case
    """
    model = TestDetectionStaticModel(model='retinanet_r50_fpn_1x', yaml='configs/retinanet_r50_fpn_1x.yml')
    model.test_detection_train()

def test_faceboxes():
    """
    faceboxes test case
    """
    model = TestDetectionStaticModel(model='faceboxes', yaml='configs/face_detection/faceboxes.yml')
    model.test_detection_train()

