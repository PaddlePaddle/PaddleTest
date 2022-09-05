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


from ModelsTestFramework import RepoInit
from ModelsTestFramework import RepoDataset
from ModelsTestFramework import TestOcrModelFunction


def get_model_list(filename="models_list.yaml"):
    """
    get_model_list
    """
    import sys

    result = []
    with open(filename) as f:
        lines = f.readlines()
        for line in lines:
            result.append(line.strip("\n"))
    return result


def setup_module():
    """
    ocr_setup
    """
    RepoInit(repo="PaddleOCR")
    RepoDataset()


@allure.story("paddle_ocr_cli")
@pytest.mark.parametrize("cmd", get_model_list("ocr_cli_list.txt"))
def test_Speech_accuracy_cli(cmd):
    """
    test_Speech_accuracy_cli
    """
    allure.dynamic.title("paddle_ocr_cli")
    allure.dynamic.description("paddle_ocr_cli")

    model = TestOcrModelFunction()
    print(cmd)
    model.test_ocr_cli(cmd)


@allure.story("get_pretrained_model")
@pytest.mark.parametrize("yml_name", get_model_list())
def test_ocr_accuracy_get_pretrained_model(yml_name):
    """
    test_ocr_accuracy_get_pretrained_model
    """
    model_name = os.path.splitext(os.path.basename(yml_name))[0]
    allure.dynamic.title(model_name + "_get_pretrained_model")
    allure.dynamic.description("获取预训练模型")
    r = re.search("/(.*)/", yml_name)
    category = r.group(1)
    print(category)
    model = TestOcrModelFunction(model=model_name, yml=yml_name, category=category)
    model.test_ocr_get_pretrained_model()


@allure.story("eval")
@pytest.mark.parametrize("yml_name", get_model_list())
@pytest.mark.parametrize("use_gpu", [True])
def test_ocr_accuracy_eval(yml_name, use_gpu):
    """
    test_ocr_accuracy_eval
    """
    model_name = os.path.splitext(os.path.basename(yml_name))[0]
    if use_gpu is True:
        hardware = "_GPU"
    else:
        hardware = "_CPU"
    allure.dynamic.title(model_name + hardware + "_eval")
    allure.dynamic.description("模型评估")
    r = re.search("/(.*)/", yml_name)
    r = re.search("/(.*)/", yml_name)
    category = r.group(1)
    print(category)
    model = TestOcrModelFunction(model=model_name, yml=yml_name, category=category)
    model.test_ocr_eval(use_gpu)


@allure.story("infer")
@pytest.mark.parametrize("yml_name", get_model_list())
@pytest.mark.parametrize("use_gpu", [True, False])
def test_ocr_accuracy_infer(yml_name, use_gpu):
    """
    test_ocr_accuracy_infer
    """
    model_name = os.path.splitext(os.path.basename(yml_name))[0]
    if use_gpu is True:
        hardware = "_GPU"
    else:
        hardware = "_CPU"
    allure.dynamic.title(model_name + hardware + "_infer")
    allure.dynamic.description("模型预测")
    r = re.search("/(.*)/", yml_name)
    category = r.group(1)
    print(category)
    model = TestOcrModelFunction(model=model_name, yml=yml_name, category=category)
    model.test_ocr_rec_infer(use_gpu)


@allure.story("export_model")
@pytest.mark.parametrize("yml_name", get_model_list())
@pytest.mark.parametrize("use_gpu", [True, False])
def test_ocr_accuracy_export_model(yml_name, use_gpu):
    """
    test_ocr_accuracy_export_model
    """
    model_name = os.path.splitext(os.path.basename(yml_name))[0]
    if use_gpu is True:
        hardware = "_GPU"
    else:
        hardware = "_CPU"
    allure.dynamic.title(model_name + hardware + "_export_model")
    allure.dynamic.description("模型动转静")

    if model_name == "re_vi_layoutxlm_xfund_zh":
        pytest.skip("not supported")

    r = re.search("/(.*)/", yml_name)
    category = r.group(1)
    print(category)
    model = TestOcrModelFunction(model=model_name, yml=yml_name, category=category)
    model.test_ocr_export_model(use_gpu)


@allure.story("predict")
@pytest.mark.parametrize("yml_name", get_model_list())
@pytest.mark.parametrize("enable_mkldnn", [True, False])
def test_ocr_accuracy_predict_mkl(yml_name, enable_mkldnn):
    """
    test_ocr_accuracy_predict_mkl
    """
    model_name = os.path.splitext(os.path.basename(yml_name))[0]
    if enable_mkldnn is True:
        hardware = "_mkldnn"
    else:
        hardware = "_CPU"
    allure.dynamic.title(model_name + hardware + "_predict")
    allure.dynamic.description("预测库预测")

    if model_name == "re_vi_layoutxlm_xfund_zh":
        pytest.skip("not supported")

    r = re.search("/(.*)/", yml_name)
    category = r.group(1)
    print(category)
    model = TestOcrModelFunction(model=model_name, yml=yml_name, category=category)
    model.test_ocr_rec_predict(False, 0, enable_mkldnn)


@allure.story("predict")
@pytest.mark.parametrize("yml_name", get_model_list())
@pytest.mark.parametrize("use_tensorrt", [True, False])
def test_ocr_accuracy_predict_trt(yml_name, use_tensorrt):
    """
    test_ocr_accuracy_predict_trt
    """
    model_name = os.path.splitext(os.path.basename(yml_name))[0]
    if use_tensorrt is True:
        hardware = "_tensorRT"
    else:
        hardware = "_GPU"
    allure.dynamic.title(model_name + hardware + "_predict")
    allure.dynamic.description("预测库预测")

    if model_name == "re_vi_layoutxlm_xfund_zh":
        pytest.skip("not supported")

    r = re.search("/(.*)/", yml_name)
    category = r.group(1)
    print(category)
    model = TestOcrModelFunction(model=model_name, yml=yml_name, category=category)
    model.test_ocr_rec_predict(True, use_tensorrt, 0)


def test_ocr_accuracy_predict_recovery():
    """
    test_ocr_accuracy_predict_recovery
    """
    hardware = "_GPU"
    allure.dynamic.title("layout" + hardware + "_predict_recovery")
    allure.dynamic.description("预测库预测")

    model = TestOcrModelFunction()
    model.test_ocr_predict_recovery(True)


@allure.story("train")
@pytest.mark.parametrize("yml_name", get_model_list())
@pytest.mark.parametrize("use_gpu", [True])
def test_ocr_funtion_train(yml_name, use_gpu):
    """
    test_ocr_funtion_train
    """
    model_name = os.path.splitext(os.path.basename(yml_name))[0]
    if use_gpu is True:
        hardware = "_GPU"
    else:
        hardware = "_CPU"
    allure.dynamic.title(model_name + hardware + "_train")
    allure.dynamic.description("训练")
    r = re.search("/(.*)/", yml_name)
    category = r.group(1)
    print(category)
    model = TestOcrModelFunction(model=model_name, yml=yml_name, category=category)
    model.test_ocr_train(use_gpu)
