# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test resnet50 model
"""
import os
import sys
import logging
import six

import wget
import pytest
import numpy as np

# pylint: disable=wrong-import-position
sys.path.append("..")
from test_case import InferenceTest

# pylint: enable=wrong-import-position


def check_model_exist():
    """
    check model exist
    """
    resnet50_url = "https://paddle-inference-dist.bj.bcebos.com/Paddle-Inference-Demo/resnet50.tgz"
    if not os.path.exists("./resnet50/inference.pdiparams"):
        wget.download(resnet50_url, out="./")


@pytest.mark.p0
@pytest.mark.config_init_combined_model
def test_config():
    """
    test combined model config
    """
    check_model_exist()
    test_suite = InferenceTest()
    test_suite.load_config(model_file="./resnet50/inference.pdmodel", params_file="./resnet50/inference.pdiparams")
    test_suite.config_test()


@pytest.mark.p0
@pytest.mark.config_disablegpu_memory
def test_disable_gpu():
    """
    test no gpu resources occupied after disable gpu
    """
    check_model_exist()
    test_suite = InferenceTest()
    test_suite.load_config(model_file="./resnet50/inference.pdmodel", params_file="./resnet50/inference.pdiparams")
    batch_size = 1
    fake_input = np.random.randn(batch_size, 3, 224, 224).astype("float32")
    input_data_dict = {"inputs": fake_input}
    test_suite.disable_gpu_test(input_data_dict)


@pytest.mark.p1
@pytest.mark.trt_fp32_bz1_precision
def test_trtfp32_bz1():
    """
    compared trt fp32 batch_size=1 resnet50 outputs with no ir config
    """
    check_model_exist()

    batch_size = 1
    fake_input = np.random.randn(batch_size, 3, 224, 224).astype("float32")
    input_data_dict = {"inputs": fake_input}

    test_suite = InferenceTest()
    test_suite.load_config(model_file="./resnet50/inference.pdmodel", params_file="./resnet50/inference.pdiparams")
    output_data_dict = test_suite.get_truth_val(input_data_dict, device="gpu")

    del test_suite  # destroy class to save memory

    test_suite2 = InferenceTest()
    test_suite2.load_config(model_file="./resnet50/inference.pdmodel", params_file="./resnet50/inference.pdiparams")
    test_suite2.trt_fp32_bz1_test(input_data_dict, output_data_dict)
