# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test pcpvt_base model
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
    tnt_small_url = "https://paddle-qa.bj.bcebos.com/inference_model/2.1.1/class/pcpvt_base.tgz"
    if not os.path.exists("./pcpvt_base/inference.pdiparams"):
        wget.download(tnt_small_url, out="./")
        tar = tarfile.open("pcpvt_base.tgz")
        tar.extractall()
        tar.close()


@pytest.mark.p0
@pytest.mark.jetson
@pytest.mark.config_init_combined_model
def test_config():
    """
    test combined model config
    """
    check_model_exist()
    test_suite = InferenceTest()
    test_suite.load_config(model_file="./pcpvt_base/inference.pdmodel", params_file="./pcpvt_base/inference.pdiparams")
    test_suite.config_test()


@pytest.mark.p0
@pytest.mark.config_disablegpu_memory
def test_disable_gpu():
    """
    test no gpu resources occupied after disable gpu
    """
    check_model_exist()
    test_suite = InferenceTest()
    test_suite.load_config(model_file="./pcpvt_base/inference.pdmodel", params_file="./pcpvt_base/inference.pdiparams")
    batch_size = 1
    fake_input = np.random.randn(batch_size, 3, 224, 224).astype("float32")
    input_data_dict = {"x": fake_input}
    test_suite.disable_gpu_test(input_data_dict)


@pytest.mark.p1
@pytest.mark.jetson
@pytest.mark.trt_fp32_more_bz_precision
def test_trt_fp32_more_bz():
    """
    compared trt fp32 batch_size=1-10 pcpvt_base outputs with true val
    """
    check_model_exist()

    file_path = "./pcpvt_base"
    images_size = 224
    batch_size_pool = [1, 5]
    for batch_size in batch_size_pool:
        test_suite = InferenceTest()
        test_suite.load_config(
            model_file="./pcpvt_base/inference.pdmodel", params_file="./pcpvt_base/inference.pdiparams"
        )
        images_list, npy_list = test_suite.get_images_npy(file_path, images_size)
        fake_input = np.array(images_list[0:batch_size]).astype("float32")
        input_data_dict = {"x": fake_input}
        output_data_dict = test_suite.get_truth_val(input_data_dict, device="gpu")

        del test_suite  # destroy class to save memory

        test_suite2 = InferenceTest()
        test_suite2.load_config(
            model_file="./pcpvt_base/inference.pdmodel", params_file="./pcpvt_base/inference.pdiparams"
        )
        test_suite2.trt_more_bz_test(
            input_data_dict, output_data_dict, max_batch_size=10, precision="trt_fp32", delta=1e-2
        )

        del test_suite2  # destroy class to save memory


@pytest.mark.p1
@pytest.mark.jetson
@pytest.mark.trt_fp32_more_bz_precision
def test_jetson_trt_fp32_more_bz():
    """
    compared trt fp32 batch_size=1-10 pcpvt_base outputs with true val
    """
    check_model_exist()

    file_path = "./pcpvt_base"
    images_size = 224
    batch_size_pool = [1, 5 ,10]
    for batch_size in batch_size_pool:
        test_suite = InferenceTest()
        test_suite.load_config(
            model_file="./pcpvt_base/inference.pdmodel", params_file="./pcpvt_base/inference.pdiparams"
        )
        images_list, npy_list = test_suite.get_images_npy(file_path, images_size)
        fake_input = np.array(images_list[0:batch_size]).astype("float32")
        input_data_dict = {"x": fake_input}
        output_data_dict = test_suite.get_truth_val(input_data_dict, device="gpu")

        del test_suite  # destroy class to save memory

        test_suite2 = InferenceTest()
        test_suite2.load_config(
            model_file="./pcpvt_base/inference.pdmodel", params_file="./pcpvt_base/inference.pdiparams"
        )
        test_suite2.trt_more_bz_test(
            input_data_dict, output_data_dict, max_batch_size=10, precision="trt_fp32", delta=4e-1
        )

        del test_suite2  # destroy class to save memory

@pytest.mark.p1
@pytest.mark.jetson
@pytest.mark.trt_fp32_multi_thread_bz1_precision
def test_trt_fp32_bz1_multi_thread():
    """
    compared trt fp32 batch_size=1 pcpvt_base multi_thread outputs with true val
    """
    check_model_exist()

    file_path = "./pcpvt_base"
    images_size = 224
    batch_size = 1
    test_suite = InferenceTest()
    test_suite.load_config(model_file="./pcpvt_base/inference.pdmodel", params_file="./pcpvt_base/inference.pdiparams")
    images_list, npy_list = test_suite.get_images_npy(file_path, images_size)
    fake_input = np.array(images_list[0:batch_size]).astype("float32")
    input_data_dict = {"x": fake_input}
    output_data_dict = test_suite.get_truth_val(input_data_dict, device="gpu")

    del test_suite  # destroy class to save memory

    test_suite2 = InferenceTest()
    test_suite2.load_config(model_file="./pcpvt_base/inference.pdmodel", params_file="./pcpvt_base/inference.pdiparams")
    test_suite2.trt_bz1_multi_thread_test(input_data_dict, output_data_dict, delta=1e-2, precision="trt_fp32")

    del test_suite2  # destroy class to save memory
