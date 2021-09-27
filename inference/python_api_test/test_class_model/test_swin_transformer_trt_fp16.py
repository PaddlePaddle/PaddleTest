# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test swin_transformer model
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
    swin_transformer_url = "https://paddle-qa.bj.bcebos.com/inference_model/2.1/class/swin_transformer.tgz"
    if not os.path.exists("./swin_transformer/inference.pdiparams"):
        wget.download(swin_transformer_url, out="./")
        tar = tarfile.open("swin_transformer.tgz")
        tar.extractall()
        tar.close()


def test_config():
    """
    test combined model config
    """
    check_model_exist()
    test_suite = InferenceTest()
    test_suite.load_config(
        model_file="./swin_transformer/inference.pdmodel", params_file="./swin_transformer/inference.pdiparams"
    )
    test_suite.config_test()


@pytest.mark.server
@pytest.mark.trt_fp16
def test_trt_fp16_more_bz():
    """
    compared trt fp32 batch_size=1-10 swin_transformer outputs with true val
    """
    check_model_exist()

    file_path = "./swin_transformer"
    images_size = 384
    batch_size_pool = [1, 5]
    for batch_size in batch_size_pool:
        test_suite = InferenceTest()
        test_suite.load_config(
            model_file="./swin_transformer/inference.pdmodel", params_file="./swin_transformer/inference.pdiparams"
        )
        images_list, npy_list = test_suite.get_images_npy(file_path, images_size)
        fake_input = np.array(images_list[0:batch_size]).astype("float32")
        input_data_dict = {"x": fake_input}
        output_data_dict = test_suite.get_truth_val(input_data_dict, device="gpu")

        del test_suite  # destroy class to save memory

        test_suite2 = InferenceTest()
        test_suite2.load_config(
            model_file="./swin_transformer/inference.pdmodel", params_file="./swin_transformer/inference.pdiparams"
        )
        test_suite2.trt_more_bz_test(
            input_data_dict,
            output_data_dict,
            repeat=1,
            delta=1e-4,
            min_subgraph_size=5,
            precision="trt_fp16",
            max_batch_size=batch_size,
        )

        del test_suite2  # destroy class to save memory


@pytest.mark.jetson
@pytest.mark.trt_fp16
def test_jetson_trt_fp16_more_bz():
    """
    compared trt fp32 batch_size=1-10 swin_transformer outputs with true val
    """
    check_model_exist()

    file_path = "./swin_transformer"
    images_size = 384
    batch_size_pool = [1]
    for batch_size in batch_size_pool:
        test_suite = InferenceTest()
        test_suite.load_config(
            model_file="./swin_transformer/inference.pdmodel", params_file="./swin_transformer/inference.pdiparams"
        )
        images_list, npy_list = test_suite.get_images_npy(file_path, images_size)
        fake_input = np.array(images_list[0:batch_size]).astype("float32")
        input_data_dict = {"x": fake_input}
        output_data_dict = test_suite.get_truth_val(input_data_dict, device="gpu")

        del test_suite  # destroy class to save memory

        test_suite2 = InferenceTest()
        test_suite2.load_config(
            model_file="./swin_transformer/inference.pdmodel", params_file="./swin_transformer/inference.pdiparams"
        )
        test_suite2.trt_more_bz_test(
            input_data_dict,
            output_data_dict,
            repeat=1,
            delta=1e-4,
            min_subgraph_size=1,
            precision="trt_fp16",
            max_batch_size=batch_size,
        )

        del test_suite2  # destroy class to save memory


@pytest.mark.server
@pytest.mark.trt_fp16_multi_thread
def test_trt_fp16_bz1_multi_thread():
    """
    compared trt fp32 batch_size=1 swin_transformer multi_thread outputs with true val
    """
    check_model_exist()

    file_path = "./swin_transformer"
    images_size = 384
    batch_size = 1
    test_suite = InferenceTest()
    test_suite.load_config(
        model_file="./swin_transformer/inference.pdmodel", params_file="./swin_transformer/inference.pdiparams"
    )
    images_list, npy_list = test_suite.get_images_npy(file_path, images_size)
    fake_input = np.array(images_list[0:batch_size]).astype("float32")
    input_data_dict = {"x": fake_input}
    output_data_dict = test_suite.get_truth_val(input_data_dict, device="gpu")

    del test_suite  # destroy class to save memory

    test_suite2 = InferenceTest()
    test_suite2.load_config(
        model_file="./swin_transformer/inference.pdmodel", params_file="./swin_transformer/inference.pdiparams"
    )
    test_suite2.trt_bz1_multi_thread_test(
        input_data_dict, output_data_dict, repeat=1, delta=1e-4, min_subgraph_size=1, precision="trt_fp16"
    )

    del test_suite2  # destroy class to save memory
