# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test fcn_hrnetw18 model
"""

import os
import sys
import logging
import tarfile
import shutil
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
    fcn_hrnetw18_url = "https://paddle-qa.bj.bcebos.com/inference_model/2.6/seg/fcn_hrnetw18.tgz"
    if not os.path.exists("./fcn_hrnetw18/model.pdiparams"):
        wget.download(fcn_hrnetw18_url, out="./")
        tar = tarfile.open("fcn_hrnetw18.tgz")
        tar.extractall()
        tar.close()


def test_config():
    """
    test combined model config
    """
    check_model_exist()
    test_suite = InferenceTest()
    test_suite.load_config(
        model_file="./fcn_hrnetw18/model.pdmodel",
        params_file="./fcn_hrnetw18/model.pdiparams",
    )
    test_suite.config_test()


@pytest.mark.win
@pytest.mark.server
@pytest.mark.trt_fp32
def test_trt_fp32_more_bz():
    """
    compared trt fp32 batch_size=1 fcn_hrnetw18 outputs with true val
    """
    check_model_exist()

    file_path = "./fcn_hrnetw18"
    images_size = 224
    batch_size_pool = [1]
    max_batch_size = 2
    for batch_size in batch_size_pool:

        test_suite = InferenceTest()
        test_suite.load_config(
            model_file="./fcn_hrnetw18/model.pdmodel",
            params_file="./fcn_hrnetw18/model.pdiparams",
        )
        images_list, npy_list = test_suite.get_images_npy(file_path, images_size)
        fake_input = np.array(images_list[0:batch_size]).astype("float32")
        input_data_dict = {"x": fake_input}
        output_data_dict = test_suite.get_truth_val(input_data_dict, device="gpu")

        del test_suite  # destroy class to save memory

        test_suite2 = InferenceTest()
        test_suite2.load_config(
            model_file="./fcn_hrnetw18/model.pdmodel",
            params_file="./fcn_hrnetw18/model.pdiparams",
        )
        test_suite2.trt_more_bz_test(
            input_data_dict,
            output_data_dict,
            max_batch_size=max_batch_size,
            min_subgraph_size=1,
            precision="trt_fp32",
            dynamic=True,
            auto_tuned=True,
        )

        del test_suite2  # destroy class to save memory


@pytest.mark.jetson
@pytest.mark.trt_fp32_more_bz_precision
def test_jetson_trt_fp32_more_bz():
    """
    compared trt fp32 batch_size=1 fcn_hrnetw18 outputs with true val
    """
    check_model_exist()

    file_path = "./fcn_hrnetw18"
    images_size = 224
    batch_size_pool = [1]
    max_batch_size = 1
    for batch_size in batch_size_pool:

        test_suite = InferenceTest()
        test_suite.load_config(
            model_file="./fcn_hrnetw18/model.pdmodel",
            params_file="./fcn_hrnetw18/model.pdiparams",
        )
        images_list, npy_list = test_suite.get_images_npy(file_path, images_size)
        fake_input = np.array(images_list[0:batch_size]).astype("float32")
        input_data_dict = {"x": fake_input}
        output_data_dict = test_suite.get_truth_val(input_data_dict, device="gpu")

        del test_suite  # destroy class to save memory

        test_suite2 = InferenceTest()
        test_suite2.load_config(
            model_file="./fcn_hrnetw18/model.pdmodel",
            params_file="./fcn_hrnetw18/model.pdiparams",
        )
        test_suite2.trt_more_bz_test(
            input_data_dict,
            output_data_dict,
            max_batch_size=max_batch_size,
            min_subgraph_size=1,
            precision="trt_fp32",
            dynamic=True,
            auto_tuned=True,
        )

        del test_suite2  # destroy class to save memory


@pytest.mark.trt_fp32_multi_thread
def test_trt_fp32_bz1_multi_thread():
    """
    compared trt fp32 batch_size=1 fcn_hrnetw18 multi_thread outputs with true val
    """
    check_model_exist()

    file_path = "./fcn_hrnetw18"
    images_size = 224
    batch_size = 1
    test_suite = InferenceTest()
    test_suite.load_config(
        model_file="./fcn_hrnetw18/model.pdmodel",
        params_file="./fcn_hrnetw18/model.pdiparams",
    )
    images_list, npy_list = test_suite.get_images_npy(file_path, images_size)
    fake_input = np.array(images_list[0:batch_size]).astype("float32")
    input_data_dict = {"x": fake_input}
    output_data_dict = test_suite.get_truth_val(input_data_dict, device="gpu")

    del test_suite  # destroy class to save memory

    test_suite2 = InferenceTest()
    test_suite2.load_config(
        model_file="./fcn_hrnetw18/model.pdmodel",
        params_file="./fcn_hrnetw18/model.pdiparams",
    )
    test_suite2.trt_bz1_multi_thread_test(
        input_data_dict,
        output_data_dict,
        precision="trt_fp32",
    )

    del test_suite2  # destroy class to save memory
