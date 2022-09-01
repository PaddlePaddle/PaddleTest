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
import os
import re
import allure

from RocmTestFramework import TestClassModel
from RocmTestFramework import RepoInit
from RocmTestFramework import RepoRemove
from RocmTestFramework import RepoDataset
from RocmTestFramework import clean_process
from RocmTestFramework import get_model_list

def setup_module():
    """
    """
    RepoInit(repo='PaddleClas')
    RepoDataset(cmd='cd PaddleClas; rm -rf dataset; ln -s /ssd2/ce_data/PaddleClas dataset;')

def teardown_module():
    """
    """
#    RepoRemove(repo='PaddleClas')

def setup_function():
    clean_process()


@allure.story('train')
@pytest.mark.parametrize('yml_name', get_model_list('clas_model_list.yaml'))
def test_class_funtion_train(yml_name):
    model_name=os.path.splitext(os.path.basename(yml_name))[0]
    hardware='_GPU'
    allure.dynamic.title(model_name+hardware+'_train')
    allure.dynamic.description('训练')
    model = TestClassModel(model=model_name, yaml=yml_name)
    model.test_class_train()

@allure.story('get_pretrained_model')
@pytest.mark.parametrize('yml_name', get_model_list('clas_model_list.yaml'))
def test_class_accuracy_get_pretrained_model(yml_name):
    model_name=os.path.splitext(os.path.basename(yml_name))[0]
    allure.dynamic.title(model_name+'_get_pretrained_model')
    allure.dynamic.description('获取预训练模型')
    model = TestClassModel(model=model_name, yaml=yml_name)
    model.test_class_get_pretrained_model()


@allure.story('export_model')
@pytest.mark.parametrize('yml_name', get_model_list('clas_model_list.yaml'))
def test_class_accuracy_export_model(yml_name):
    model_name=os.path.splitext(os.path.basename(yml_name))[0]
    hardware='_GPU'
    allure.dynamic.title(model_name+hardware+'_export_model')
    allure.dynamic.description('模型动转静')

    model = TestClassModel(model=model_name, yaml=yml_name)
    model.test_class_export_model()

@allure.story('predict')
@pytest.mark.parametrize('yml_name', get_model_list('clas_model_list.yaml'))
def test_class_accuracy_predict(yml_name):
    model_name=os.path.splitext(os.path.basename(yml_name))[0]
    hardware='_GPU'
    allure.dynamic.title(model_name+hardware+'_predict')
    allure.dynamic.description('预测库预测')
    model = TestClassModel(model=model_name, yaml=yml_name)
    model.test_class_predict()


