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

from RocmTestFramework import TestSegModel
from RocmTestFramework import RepoInit
from RocmTestFramework import RepoRemove
from RocmTestFramework import RepoDataset
from RocmTestFramework import clean_process
from RocmTestFramework import get_model_list



def setup_module():
    """
    """
    RepoInit(repo='PaddleSeg')
    RepoDataset(cmd='''cd PaddleSeg; mkdir data; cd data; rm -rf cityscapes; ln -s /ssd2/ce_data/PaddleSeg/cityscape cityscapes; ln -s /ssd2/ce_data/PaddleSeg/mini_supervisely mini_supervisely''') 


def teardown_module():
    """
    """
    RepoRemove(repo='PaddleSeg')


@allure.story('train')
@pytest.mark.parametrize('yml_name', get_model_list('seg_model_list.yaml'))
def test_seg_funtion_train(yml_name):
    model_name=os.path.splitext(os.path.basename(yml_name))[0]
    hardware='_GPU'
    allure.dynamic.title(model_name+hardware+'_train')
    allure.dynamic.description('训练')
    if (model_name=='pp_humanseg_lite_mini_supervisely'):
        pytest.skip("skip, run time too long")
    model = TestSegModel(model=model_name, yaml=yml_name)
    model.test_seg_train()
