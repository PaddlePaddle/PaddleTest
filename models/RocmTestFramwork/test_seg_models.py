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

from RocmTestFramework import TestSegModel
from RocmTestFramework import RepoInit
from RocmTestFramework import RepoRemove
from RocmTestFramework import RepoDataset
from RocmTestFramework import clean_process



def setup_module():
    """
    """
    RepoInit(repo='PaddleSeg')
    RepoDataset(cmd='''cd PaddleSeg; mkdir data; cd data; rm -rf cityscapes; ln -s /data/cityscape cityscapes; cd ..''') 


def teardown_module():
    """
    """
    RepoRemove(repo='PaddleSeg')

def setup_function():
    clean_process()

def test_deeplabv3_resnet50_os8_cityscapes_1024x512_80k():
    """
    deeplabv3_resnet50_os8_cityscapes_1024x512_80k test case
    """
    model = TestSegModel(model='deeplabv3_resnet50_os8_cityscapes_1024x512_80k',
                         yaml='configs/deeplabv3/deeplabv3_resnet50_os8_cityscapes_1024x512_80k.yml')
    model.test_seg_train()

def test_fcn_hrnetw48_cityscapes_1024x512_80k():
    """
    fcn_hrnetw48_cityscapes_1024x512_80k test case
    """
    model = TestSegModel(model='fcn_hrnetw48_cityscapes_1024x512_80k',
                         yaml='configs/fcn/fcn_hrnetw48_cityscapes_1024x512_80k.yml')
    model.test_seg_train()

def test_danet_resnet50_os8_cityscapes_1024x512_80k():
    """
    danet_resnet50_os8_cityscapes_1024x512_80k test case
    """
    model = TestSegModel(model='danet_resnet50_os8_cityscapes_1024x512_80k',
                         yaml='configs/danet/danet_resnet50_os8_cityscapes_1024x512_80k.yml')
    model.test_seg_train()

def test_fastscnn_cityscapes_1024x1024_160k():
    """
    fastscnn_cityscapes_1024x1024_160k test case
    """
    model = TestSegModel(model='fastscnn_cityscapes_1024x1024_160k',
                         yaml='configs/fastscnn/fastscnn_cityscapes_1024x1024_160k.yml')
    model.test_seg_train()

def test_ocrnet_hrnetw18_cityscapes_1024x512_160k():
    """
    ocrnet_hrnetw18_cityscapes_1024x512_160k test case
    """
    model = TestSegModel(model='ocrnet_hrnetw18_cityscapes_1024x512_160k',
                         yaml='configs/ocrnet/ocrnet_hrnetw18_cityscapes_1024x512_160k.yml')
    model.test_seg_train()
   
def test_unet_cityscapes_1024x512_160k():
    """
    unet_cityscapes_1024x512_160k test case
    """
    model = TestSegModel(model='unet_cityscapes_1024x512_160k',
                         yaml='configs/unet/unet_cityscapes_1024x512_160k.yml')
    model.test_seg_train() 

def test_ann_resnet50_os8_cityscapes_1024x512_80k():
    """
    ann_resnet50_os8_cityscapes_1024x512_80k test case
    """
    model = TestSegModel(model='ann_resnet50_os8_cityscapes_1024x512_80k', 
                         yaml='configs/ann/ann_resnet50_os8_cityscapes_1024x512_80k.yml')
    model.test_seg_eval()

def test_bisenet_cityscapes_1024x1024_160k():
    """
    bisenet_cityscapes_1024x1024_160k test case
    """
    model = TestSegModel(model='bisenet_cityscapes_1024x1024_160k',
                         yaml='configs/bisenet/bisenet_cityscapes_1024x1024_160k.yml')
    model.test_seg_eval()

def test_gcnet_resnet50_os8_cityscapes_1024x512_80k():
    """
    gcnet_resnet50_os8_cityscapes_1024x512_80k test case
    """
    model = TestSegModel(model='gcnet_resnet50_os8_cityscapes_1024x512_80k',
                         yaml='configs/gcnet/gcnet_resnet50_os8_cityscapes_1024x512_80k.yml')
    model.test_seg_eval()

def test_gscnn_resnet50_os8_cityscapes_1024x512_80k():
    """
    gscnn_resnet50_os8_cityscapes_1024x512_80k test case
    """
    model = TestSegModel(model='gscnn_resnet50_os8_cityscapes_1024x512_80k',
                         yaml='configs/gscnn/gscnn_resnet50_os8_cityscapes_1024x512_80k.yml')
    model.test_seg_eval()

def test_hardnet_cityscapes_1024x1024_160k():
    """
    hardnet_cityscapes_1024x1024_160k test case
    """
    model = TestSegModel(model='hardnet_cityscapes_1024x1024_160k',
                         yaml='configs/hardnet/hardnet_cityscapes_1024x1024_160k.yml')
    model.test_seg_eval()

def test_decoupledsegnet_resnet50_os8_cityscapes_1024x512_80k():
    """
    decoupledsegnet_resnet50_os8_cityscapes_1024x512_80k test case
    """
    model = TestSegModel(model='decoupledsegnet_resnet50_os8_cityscapes_1024x512_80k',
                         yaml='configs/decoupled_segnet/decoupledsegnet_resnet50_os8_cityscapes_1024x512_80k.yml')
    model.test_seg_eval()

def test_emanet_resnet50_os8_cityscapes_1024x512_80k():
    """
    emanet_resnet50_os8_cityscapes_1024x512_80k test case
    """
    model = TestSegModel(model='emanet_resnet50_os8_cityscapes_1024x512_80k',
                         yaml='configs/emanet/emanet_resnet50_os8_cityscapes_1024x512_80k.yml')
    model.test_seg_eval()

def test_isanet_resnet50_os8_cityscapes_769x769_80k():
    """
    isanet_resnet50_os8_cityscapes_769x769_80k test case
    """
    model = TestSegModel(model='isanet_resnet50_os8_cityscapes_769x769_80k',
                         yaml='configs/isanet/isanet_resnet50_os8_cityscapes_769x769_80k.yml')
    model.test_seg_eval()

def test_dnlnet_resnet50_os8_cityscapes_1024x512_80k():
    """
    dnlnet_resnet50_os8_cityscapes_1024x512_80k test case
    """
    model = TestSegModel(model='dnlnet_resnet50_os8_cityscapes_1024x512_80k',
                         yaml='configs/dnlnet/dnlnet_resnet50_os8_cityscapes_1024x512_80k.yml')
    model.test_seg_eval()

