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
import os
import allure

from RocmTestFramework import TestDetectionDygraphModel
from RocmTestFramework import RepoInit
from RocmTestFramework import RepoRemove
from RocmTestFramework import RepoDataset
from RocmTestFramework import clean_process
from RocmTestFramework import get_model_list

def setup_module():
    """
    """
    RepoInit(repo='PaddleDetection')
    RepoDataset(cmd='''cd PaddleDetection;
                     rm -rf dataset; 
                     ln -s /ssd2/ce_data/PaddleDetection/data/ dataset; 
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


@allure.story('train')
@pytest.mark.parametrize('yml_name', get_model_list('detection_model_list.yaml'))
def test_class_funtion_train(yml_name):
    model_name=os.path.splitext(os.path.basename(yml_name))[0]
    hardware='_GPU'
    allure.dynamic.title(model_name+hardware+'_train')
    allure.dynamic.description('训练')
    model = TestDetectionDygraphModel(model=model_name, yaml=yml_name)
    model.test_detection_train()
