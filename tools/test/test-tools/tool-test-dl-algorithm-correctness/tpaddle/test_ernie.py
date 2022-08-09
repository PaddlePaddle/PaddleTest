# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test ernie model
"""

import os
import sys
import logging
import tarfile
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
    ernie_url = "https://paddle-qa.bj.bcebos.com/inference_model/2.1.2/nlp/ernie.tgz"
    if not os.path.exists("./ernie/inference.pdiparams"):
        wget.download(ernie_url, out="./")
        tar = tarfile.open("ernie.tgz")
        tar.extractall()
        tar.close()


def test_config():
    """
    test combined model config
    """
    check_model_exist()
    test_suite = InferenceTest()
    test_suite.load_config(model_file="./ernie/inference.pdmodel", params_file="./ernie/inference.pdiparams")
    test_suite.config_test()


@pytest.mark.win
@pytest.mark.server
@pytest.mark.config_disablegpu_memory
def test_disable_gpu():
    """
    test no gpu resources occupied after disable gpu
    """
    check_model_exist()
    test_suite = InferenceTest()
    test_suite.load_config(model_file="./ernie/inference.pdmodel", params_file="./ernie/inference.pdiparams")
    batch_size = 1
    fake_input0 = np.zeros((batch_size, 128)).astype("int64")
    fake_input1 = np.zeros((batch_size, 128)).astype("int64")
    input_data_dict = {"input_ids": fake_input0, "token_type_ids": fake_input1}
    test_suite.disable_gpu_test(input_data_dict)


@pytest.mark.win
@pytest.mark.server
@pytest.mark.jetson
@pytest.mark.gpu
def test_gpu_bz1():
    """
    compared gpu bert outputs with true val
    """
    check_model_exist()

    test_suite = InferenceTest()
    test_suite.load_config(model_file="./ernie/inference.pdmodel", params_file="./ernie/inference.pdiparams")
    data_path = "./ernie/data.txt"
    images_list = test_suite.get_text_npy(data_path)

    input_data_dict = {
        "input_ids": np.array([images_list[0][0]]).astype("int64"),
        "token_type_ids": np.array([images_list[0][1]]).astype("int64"),
    }
    output_data_dict = test_suite.get_truth_val(input_data_dict, device="cpu")

    del test_suite  # destroy class to save memory

    test_suite2 = InferenceTest()
    test_suite2.load_config(model_file="./ernie/inference.pdmodel", params_file="./ernie/inference.pdiparams")
    test_suite2.gpu_more_bz_test(input_data_dict, output_data_dict, delta=1e-5)

    del test_suite2  # destroy class to save memory


@pytest.mark.win
@pytest.mark.server
@pytest.mark.mkldnn
def test_mkldnn():
    """
    compared mkldnn bert outputs with true val
    """
    check_model_exist()

    test_suite = InferenceTest()
    test_suite.load_config(model_file="./ernie/inference.pdmodel", params_file="./ernie/inference.pdiparams")
    data_path = "./ernie/data.txt"
    images_list = test_suite.get_text_npy(data_path)

    input_data_dict = {
        "input_ids": np.array([images_list[0][0]]).astype("int64"),
        "token_type_ids": np.array([images_list[0][1]]).astype("int64"),
    }
    output_data_dict = test_suite.get_truth_val(input_data_dict, device="cpu")

    del test_suite  # destroy class to save memory

    test_suite2 = InferenceTest()
    test_suite2.load_config(model_file="./ernie/inference.pdmodel", params_file="./ernie/inference.pdiparams")
    test_suite2.mkldnn_test(input_data_dict, output_data_dict, delta=1e-5)

    del test_suite2  # destroy class to save memory


@pytest.mark.win
@pytest.mark.server
@pytest.mark.jetson
@pytest.mark.trt_fp16
def test_trt_fp16_bz1():
    """
    compared trt fp16 batch_size=1 ernie outputs with true val
    """
    check_model_exist()

    test_suite = InferenceTest()
    test_suite.load_config(model_file="./ernie/inference.pdmodel", params_file="./ernie/inference.pdiparams")
    data_path = "./ernie/data.txt"
    images_list = test_suite.get_text_npy(data_path)

    input_data_dict = {
        "input_ids": np.array([images_list[0][0]]).astype("int64"),
        "token_type_ids": np.array([images_list[0][1]]).astype("int64"),
    }
    output_data_dict = test_suite.get_truth_val(input_data_dict, device="cpu")

    del test_suite  # destroy class to save memory

    test_suite2 = InferenceTest()
    test_suite2.load_config(model_file="./ernie/inference.pdmodel", params_file="./ernie/inference.pdiparams")
    test_suite2.trt_more_bz_test(input_data_dict, output_data_dict, delta=1e-5, max_batch_size=1, precision="trt_fp16")

    del test_suite2  # destroy class to save memory


@pytest.mark.win
@pytest.mark.server
@pytest.mark.trt_fp16_multi_thread
def test_trt_fp16_bz1_multi_thread():
    """
    compared trt fp16 batch_size=1 multi_thread ernie outputs with true val
    """
    check_model_exist()

    test_suite = InferenceTest()
    test_suite.load_config(model_file="./ernie/inference.pdmodel", params_file="./ernie/inference.pdiparams")
    data_path = "./ernie/data.txt"
    images_list = test_suite.get_text_npy(data_path)

    input_data_dict = {
        "input_ids": np.array([images_list[0][0]]).astype("int64"),
        "token_type_ids": np.array([images_list[0][1]]).astype("int64"),
    }
    output_data_dict = test_suite.get_truth_val(input_data_dict, device="cpu")

    del test_suite  # destroy class to save memory

    test_suite2 = InferenceTest()
    test_suite2.load_config(model_file="./ernie/inference.pdmodel", params_file="./ernie/inference.pdiparams")
    test_suite2.trt_bz1_multi_thread_test(input_data_dict, output_data_dict, delta=1e-5, precision="trt_fp16")

    del test_suite2  # destroy class to save memory


@pytest.mark.win
@pytest.mark.server
@pytest.mark.jetson
@pytest.mark.trt_fp32
def test_trt_fp32_bz1():
    """
    compared trt fp32 batch_size=1 ernie outputs with true val
    """
    check_model_exist()

    test_suite = InferenceTest()
    test_suite.load_config(model_file="./ernie/inference.pdmodel", params_file="./ernie/inference.pdiparams")
    data_path = "./ernie/data.txt"
    images_list = test_suite.get_text_npy(data_path)

    input_data_dict = {
        "input_ids": np.array([images_list[0][0]]).astype("int64"),
        "token_type_ids": np.array([images_list[0][1]]).astype("int64"),
    }
    output_data_dict = test_suite.get_truth_val(input_data_dict, device="cpu")

    del test_suite  # destroy class to save memory

    test_suite2 = InferenceTest()
    test_suite2.load_config(model_file="./ernie/inference.pdmodel", params_file="./ernie/inference.pdiparams")
    test_suite2.trt_more_bz_test(input_data_dict, output_data_dict, delta=1e-5, max_batch_size=1, precision="trt_fp32")

    del test_suite2  # destroy class to save memory


@pytest.mark.win
@pytest.mark.server
@pytest.mark.trt_fp32_multi_thread
def test_trt_fp32_bz1_multi_thread():
    """
    compared trt fp32 batch_size=1 multi_thread ernie outputs with true val
    """
    check_model_exist()

    test_suite = InferenceTest()
    test_suite.load_config(model_file="./ernie/inference.pdmodel", params_file="./ernie/inference.pdiparams")
    data_path = "./ernie/data.txt"
    images_list = test_suite.get_text_npy(data_path)

    input_data_dict = {
        "input_ids": np.array([images_list[0][0]]).astype("int64"),
        "token_type_ids": np.array([images_list[0][1]]).astype("int64"),
    }
    output_data_dict = test_suite.get_truth_val(input_data_dict, device="cpu")

    del test_suite  # destroy class to save memory

    test_suite2 = InferenceTest()
    test_suite2.load_config(model_file="./ernie/inference.pdmodel", params_file="./ernie/inference.pdiparams")
    test_suite2.trt_bz1_multi_thread_test(input_data_dict, output_data_dict, delta=1e-5, precision="trt_fp32")

    del test_suite2  # destroy class to save memory
