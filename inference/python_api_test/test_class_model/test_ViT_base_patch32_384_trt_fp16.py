# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test ViT_base_patch32_384 model
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
from test_case import InferenceTest, clip_model_extra_op

# pylint: enable=wrong-import-position


def check_model_exist():
    """
    check model exist
    """
    ViT_base_patch32_384_url = (
        "https://paddle-qa.bj.bcebos.com/inference_model/unknown/vit_models/ViT_base_patch32_384.tgz"
    )
    if not os.path.exists("./ViT_base_patch32_384/inference.pdiparams"):
        wget.download(ViT_base_patch32_384_url, out="./")
        tar = tarfile.open("ViT_base_patch32_384.tgz")
        tar.extractall()
        tar.close()
        clip_model_extra_op(
            path_prefix="./ViT_base_patch32_384/inference", output_model_path="./ViT_base_patch32_384/inference"
        )


def test_config():
    """
    test combined model config
    """
    check_model_exist()
    test_suite = InferenceTest()
    test_suite.load_config(
        model_file="./ViT_base_patch32_384/inference.pdmodel", params_file="./ViT_base_patch32_384/inference.pdiparams"
    )
    test_suite.config_test()


@pytest.mark.win
@pytest.mark.server
@pytest.mark.trt_fp16
def test_trt_fp16_more_bz():
    """
    compared trt fp16 batch_size=1-10 ViT_base_patch32_384 outputs with true val
    """
    check_model_exist()

    file_path = "./ViT_base_patch32_384"
    images_size = 384
    batch_size_pool = [1]
    max_batch_size = 1
    for batch_size in batch_size_pool:
        test_suite = InferenceTest()
        test_suite.load_config(
            model_file="./ViT_base_patch32_384/inference.pdmodel",
            params_file="./ViT_base_patch32_384/inference.pdiparams",
        )
        images_list, npy_list = test_suite.get_images_npy(file_path, images_size)
        fake_input = np.array(images_list[0:batch_size]).astype("float32")
        input_data_dict = {"x": fake_input}
        output_data_dict = test_suite.get_truth_val(input_data_dict, device="gpu")

        del test_suite  # destroy class to save memory

        test_suite2 = InferenceTest()
        test_suite2.load_config(
            model_file="./ViT_base_patch32_384/inference.pdmodel",
            params_file="./ViT_base_patch32_384/inference.pdiparams",
        )
        test_suite2.trt_more_bz_test(
            input_data_dict,
            output_data_dict,
            max_batch_size=max_batch_size,
            precision="trt_fp16",
            delete_pass_list=["preln_residual_bias_fuse_pass"],
        )

        del test_suite2  # destroy class to save memory


@pytest.mark.jetson
@pytest.mark.trt_fp16_more_bz_precision
def test_jetson_trt_fp16_more_bz():
    """
    compared trt fp16 batch_size=1-10 ViT_base_patch32_384 outputs with true val
    """
    check_model_exist()

    file_path = "./ViT_base_patch32_384"
    images_size = 384
    batch_size_pool = [1, 2]
    max_batch_size = 2
    for batch_size in batch_size_pool:
        test_suite = InferenceTest()
        test_suite.load_config(
            model_file="./ViT_base_patch32_384/inference.pdmodel",
            params_file="./ViT_base_patch32_384/inference.pdiparams",
        )
        images_list, npy_list = test_suite.get_images_npy(file_path, images_size)
        fake_input = np.array(images_list[0:batch_size]).astype("float32")
        input_data_dict = {"x": fake_input}
        output_data_dict = test_suite.get_truth_val(input_data_dict, device="gpu")

        del test_suite  # destroy class to save memory

        test_suite2 = InferenceTest()
        test_suite2.load_config(
            model_file="./ViT_base_patch32_384/inference.pdmodel",
            params_file="./ViT_base_patch32_384/inference.pdiparams",
        )
        test_suite2.trt_more_bz_test(
            input_data_dict,
            output_data_dict,
            max_batch_size=max_batch_size,
            precision="trt_fp16",
            delete_pass_list=["preln_residual_bias_fuse_pass"],
        )

        del test_suite2  # destroy class to save memory


@pytest.mark.trt_fp16_multi_thread
def test_trt_fp16_bz1_multi_thread():
    """
    compared trt fp16 batch_size=1 ViT_base_patch32_384 multi_thread outputs with true val
    """
    check_model_exist()

    file_path = "./ViT_base_patch32_384"
    images_size = 384
    batch_size = 1
    test_suite = InferenceTest()
    test_suite.load_config(
        model_file="./ViT_base_patch32_384/inference.pdmodel", params_file="./ViT_base_patch32_384/inference.pdiparams"
    )
    images_list, npy_list = test_suite.get_images_npy(file_path, images_size)
    fake_input = np.array(images_list[0:batch_size]).astype("float32")
    input_data_dict = {"x": fake_input}
    output_data_dict = test_suite.get_truth_val(input_data_dict, device="gpu")

    del test_suite  # destroy class to save memory
    test_suite2 = InferenceTest()
    test_suite2.load_config(
        model_file="./ViT_base_patch32_384/inference.pdmodel", params_file="./ViT_base_patch32_384/inference.pdiparams"
    )
    test_suite2.trt_bz1_multi_thread_test(
        input_data_dict, output_data_dict, precision="trt_fp16", delete_pass_list=["preln_residual_bias_fuse_pass"]
    )

    del test_suite2  # destroy class to save memory
