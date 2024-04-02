# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
/***************************************************************************
  *
  * Copyright (c) 2022 Baidu.com, Inc. All Rights Reserved
  * @file
  * @author jiaxiao01
  * @date 2022/9/2 3:46 PM
  * @brief  ocr test case
  *
  **************************************************************************/
"""

import os
import os.path
import subprocess
import re
import sys
import platform
import random
import pytest
import numpy as np
import yaml
import allure
import paddle


from ModelsTestFramework import RepoInit3D
from ModelsTestFramework import RepoDataset3D
from ModelsTestFramework import Test3DModelFunction


def get_model_list():
    """
    get_model_list
    """
    import sys

    result = []
    ci_flag = os.environ.get("ci_flag", 0)
    if ci_flag == "1":
        filename = "models_list_3D_CI.yaml"
        print("ci_flag=1")
    else:
        filename = "models_list_3D_CE.yaml"
        print("ci_flag=0")
    with open(filename) as f:
        lines = f.readlines()
        for line in lines:
            result.append(line.strip("\n"))
    return result


def get_category(yml_name):
    """
    get_category
    """
    r = re.search("/(.*)/", yml_name)
    category = r.group(1)
    return category


def get_hardware():
    """
    get_hardware
    """
    if paddle.is_compiled_with_cuda() is True:
        hardware = "_GPU"
    else:
        hardware = "_CPU"
    return hardware


def setup_module():
    """
    3d_setup
    """
    RepoInit3D(repo="Paddle3D")
    RepoDataset3D()


@allure.story("get_pretrained_model")
@pytest.mark.parametrize("yml_name", get_model_list())
def test_3D_accuracy_get_pretrained_model(yml_name):
    """
    test_3D_accuracy_get_pretrained_model
    """
    model_name = os.path.splitext(os.path.basename(yml_name))[0]
    allure.dynamic.title(model_name + "_get_pretrained_model")
    allure.dynamic.description("获取预训练模型")

    category = get_category(yml_name)
    model = Test3DModelFunction(model=model_name, yml=yml_name, category=category)
    model.test_3D_get_pretrained_model()


@allure.story("eval")
@pytest.mark.parametrize("yml_name", get_model_list())
@pytest.mark.parametrize("use_gpu", [True])
def test_3D_accuracy_eval(yml_name, use_gpu):
    """
    test_3D_accuracy_eval
    """
    model_name = os.path.splitext(os.path.basename(yml_name))[0]
    hardware = get_hardware()
    allure.dynamic.title(model_name + hardware + "_eval")
    allure.dynamic.description("模型评估")

    category = get_category(yml_name)
    if (category == "smoke") or (category == "centerpoint"):
        pytest.skip("not suporrted  eval when bs >1")
    if sys.platform == "darwin":
        pytest.skip("mac/windows skip eval")
    model = Test3DModelFunction(model=model_name, yml=yml_name, category=category)
    model.test_3D_eval(use_gpu)


@allure.story("eval")
@pytest.mark.parametrize("yml_name", get_model_list())
@pytest.mark.parametrize("use_gpu", [True])
def test_3D_accuracy_eval_bs1(yml_name, use_gpu):
    """
    test_3D_accuracy_eval_bs1
    """
    model_name = os.path.splitext(os.path.basename(yml_name))[0]
    hardware = get_hardware()
    allure.dynamic.title(model_name + hardware + "_eval_bs1")
    allure.dynamic.description("模型评估")

    category = get_category(yml_name)
    model = Test3DModelFunction(model=model_name, yml=yml_name, category=category)
    model.test_3D_eval_bs1(use_gpu)


@allure.story("export_model")
@pytest.mark.parametrize("yml_name", get_model_list())
@pytest.mark.parametrize("use_gpu", [True])
def test_3D_accuracy_export_model(yml_name, use_gpu):
    """
    test_3D_accuracy_export_model
    """
    model_name = os.path.splitext(os.path.basename(yml_name))[0]
    hardware = get_hardware()
    allure.dynamic.title(model_name + hardware + "_export_model")
    allure.dynamic.description("模型动转静")

    category = get_category(yml_name)
    model = Test3DModelFunction(model=model_name, yml=yml_name, category=category)
    model.test_3D_export_model(use_gpu)


@allure.story("predict")
@pytest.mark.parametrize("yml_name", get_model_list())
@pytest.mark.parametrize("use_gpu", [True])
def test_3D_accuracy_predict_python(yml_name, use_gpu):
    """
    test_3D_accuracy_predict_python
    """
    model_name = os.path.splitext(os.path.basename(yml_name))[0]
    hardware = get_hardware()
    allure.dynamic.title(model_name + hardware + "_predict")
    allure.dynamic.description("预测库python预测")

    category = get_category(yml_name)
    model = Test3DModelFunction(model=model_name, yml=yml_name, category=category)
    model.test_3D_predict_python(use_gpu, False)


@allure.story("predict")
@pytest.mark.parametrize("yml_name", get_model_list())
@pytest.mark.parametrize("use_gpu", [True])
def test_3D_accuracy_predict_python_trt(yml_name, use_gpu):
    """
    test_3D_accuracy_predict_python_trt
    """
    model_name = os.path.splitext(os.path.basename(yml_name))[0]
    hardware = "_TensorRT"
    allure.dynamic.title(model_name + hardware + "_predict")
    allure.dynamic.description("预测库python预测")
    pytest.skip("not supported for tensorRT predict")
    if paddle.is_compiled_with_cuda() is False:
        pytest.skip("CPU not supported for tensorRT predict")
    category = get_category(yml_name)
    if (category == "pointpillars") or (category == "centerpoint") or (category == "squeezesegv3"):
        pytest.skip("not supoorted for tensorRT predict")
    if sys.platform == "darwin":
        pytest.skip("mac skip tensorRT predict")

    category = get_category(yml_name)
    model = Test3DModelFunction(model=model_name, yml=yml_name, category=category)
    model.test_3D_predict_python(use_gpu, True)


@allure.story("train")
@pytest.mark.parametrize("yml_name", get_model_list())
@pytest.mark.parametrize("use_gpu", [True])
def test_3D_funtion_train(yml_name, use_gpu):
    """
    test_3D_funtion_train
    """
    model_name = os.path.splitext(os.path.basename(yml_name))[0]
    hardware = get_hardware()
    allure.dynamic.title(model_name + hardware + "_train")
    allure.dynamic.description("训练")

    category = get_category(yml_name)
    model = Test3DModelFunction(model=model_name, yml=yml_name, category=category)
    model.test_3D_train(use_gpu)
