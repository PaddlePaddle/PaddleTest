#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
/***************************************************************************
  *
  * Copyright (c) 2021 Baidu.com, Inc. All Rights Reserved
  * @file 
  * @author jiaxiao01
  * @date 2021/9/10 3:46 PM
  * @brief clas rec model inference test case
  *
  **************************************************************************/
"""

import pytest
import numpy as np
import subprocess
import re

from ModelInfrenceFramework import TestClasRecInference
from ModelInfrenceFramework import RepoInit
from ModelInfrenceFramework import RepoInstructions
from ModelInfrenceFramework import clean_process


def setup_module():
    """
    git clone repo and install dependency
    """
    RepoInit(repo='PaddleClas', branch='develop')
    RepoInstructions(cmd='''unset https_proxy; unset http_proxy;
                       cd PaddleClas/deploy; mkdir models; 
                       cd models; wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/inference/ppyolov2_r50vd_dcn_mainbody_v1.0_infer.tar;
                       tar xf ppyolov2_r50vd_dcn_mainbody_v1.0_infer.tar;
                       rm -rf *.tar
                       cd ..
                       wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/data/recognition_demo_data_v1.1.tar;
                       tar xf recognition_demo_data_v1.1.tar
                       rm -rf *.tar''')

def setup_function():
    """
    clean process
    """
    clean_process()


def test_product():
    """
    product test case
    """ 
    model = TestClasRecInference(yaml='configs/inference_product.yaml', 
                                  infer_imgs='./recognition_demo_data_v1.1/test_product/daoxiangcunjinzhubing_6.jpg')
    model.test_get_clas_rec_inference_model(model='product_ResNet50_vd_aliproduct_v1.0_infer')
    model.test_clas_rec_predict(expect_bbox=[287, 128, 497, 326], expect_rec_docs='稻香村金猪饼', 
                                 expect_rec_scores=0.8278527)
    
def test_logo():
    """
    logo test case
    """
    model = TestClasRecInference(yaml='configs/inference_logo.yaml', 
                                  infer_imgs='./recognition_demo_data_v1.1/test_logo/benz-001.jpeg')
    model.test_get_clas_rec_inference_model(model='logo_rec_ResNet50_Logo3K_v1.0_infer')
    model.test_clas_rec_predict(expect_bbox=[100, 69, 222, 187], expect_rec_docs='奔驰', expect_rec_scores=0.9920622)

def test_cartoon():
    """
    cartoon test case
    """
    model = TestClasRecInference(yaml='configs/inference_cartoon.yaml', 
                                  infer_imgs='./recognition_demo_data_v1.1/test_cartoon/aisidesi-001.jpeg')
    model.test_get_clas_rec_inference_model(model='cartoon_rec_ResNet50_iCartoon_v1.0_infer')
    model.test_clas_rec_predict(expect_bbox=[221, 54, 270, 109], expect_rec_docs='艾斯德斯', expect_rec_scores=0.81022185)

def test_vehicle():
    """
    vehicle test case
    """
    model = TestClasRecInference(yaml='configs/inference_vehicle.yaml', 
                                  infer_imgs='./recognition_demo_data_v1.1/test_vehicle/audia4-102.jpeg')
    model.test_get_clas_rec_inference_model(model='vehicle_cls_ResNet50_CompCars_v1.0_infer')
    model.test_clas_rec_predict(expect_bbox=[193, 123, 995, 520], expect_rec_docs='奥迪A4', expect_rec_scores=0.64866954)


