# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test TNT_small model
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
    tnt_small_url = "https://paddle-qa.bj.bcebos.com/inference_model_clipped/2.2rc/class/TNT_small.tgz"
    if not os.path.exists("./TNT_small/inference.pdiparams"):
        wget.download(tnt_small_url, out="./")
        tar = tarfile.open("TNT_small.tgz")
        tar.extractall()
        tar.close()


def test_config():
    """
    test combined model config
    """
    check_model_exist()
    test_suite = InferenceTest()
    test_suite.load_config(
        model_file="./TNT_small/inference.pdmodel",
        params_file="./TNT_small/inference.pdiparams",
    )
    test_suite.config_test()


@pytest.mark.win
@pytest.mark.server
@pytest.mark.trt_fp16
def test_trt_fp16_more_bz():
    """
    compared trt fp16 batch_size=1-2 TNT_small outputs with true val
    """
    check_model_exist()

    file_path = "./TNT_small"
    images_size = 224
    batch_size_pool = [1, 2]
    for batch_size in batch_size_pool:
        try:
            shutil.rmtree(f"{file_path}/_opt_cache")  # delete trt serialized cache
        except Exception as e:
            print("no need to delete trt serialized cache")

        test_suite = InferenceTest()
        test_suite.load_config(
            model_file="./TNT_small/inference.pdmodel",
            params_file="./TNT_small/inference.pdiparams",
        )
        images_list, npy_list = test_suite.get_images_npy(file_path, images_size)
        fake_input = np.array(images_list[0:batch_size]).astype("float32")
        input_data_dict = {"x": fake_input}
        output_data_dict = test_suite.get_truth_val(input_data_dict, device="gpu")

        del test_suite  # destroy class to save memory

        # collect shape for trt
        test_suite_c = InferenceTest()
        test_suite_c.load_config(
            model_file=file_path + "/inference.pdmodel",
            params_file=file_path + "/inference.pdiparams",
        )
        test_suite_c.collect_shape_info(
            model_path=file_path,
            input_data_dict=input_data_dict,
            device="gpu",
        )
        del test_suite_c  # destroy class to save memory

        test_suite2 = InferenceTest()
        test_suite2.load_config(
            model_file="./TNT_small/inference.pdmodel",
            params_file="./TNT_small/inference.pdiparams",
        )

        test_suite2.trt_more_bz_test(
            input_data_dict,
            output_data_dict,
            delta=1e-2,
            max_batch_size=10,
            min_subgraph_size=0,
            precision="trt_fp16",
            dynamic=True,
            shape_range_file=file_path + "/shape_range.pbtxt",
            delete_pass_list=["trt_skip_layernorm_fuse_pass"],
        )

        del test_suite2  # destroy class to save memory


@pytest.mark.jetson
@pytest.mark.trt_fp16
def test_jetson_trt_fp16_more_bz():
    """
    compared trt fp16 batch_size=1 TNT_small outputs with true val
    """
    check_model_exist()

    file_path = "./TNT_small"
    images_size = 224
    batch_size_pool = [1]
    for batch_size in batch_size_pool:
        test_suite = InferenceTest()
        test_suite.load_config(
            model_file="./TNT_small/inference.pdmodel",
            params_file="./TNT_small/inference.pdiparams",
        )
        images_list, npy_list = test_suite.get_images_npy(file_path, images_size)
        fake_input = np.array(images_list[0:batch_size]).astype("float32")
        input_data_dict = {"x": fake_input}
        output_data_dict = test_suite.get_truth_val(input_data_dict, device="gpu")

        del test_suite  # destroy class to save memory

        # collect shape for trt
        test_suite_c = InferenceTest()
        test_suite_c.load_config(
            model_file=file_path + "/inference.pdmodel",
            params_file=file_path + "/inference.pdiparams",
        )
        test_suite_c.collect_shape_info(
            model_path=file_path,
            input_data_dict=input_data_dict,
            device="gpu",
        )
        del test_suite_c

        test_suite2 = InferenceTest()
        test_suite2.load_config(
            model_file="./TNT_small/inference.pdmodel",
            params_file="./TNT_small/inference.pdiparams",
        )

        test_suite2.trt_more_bz_test(
            input_data_dict,
            output_data_dict,
            delta=1e-2,
            max_batch_size=10,
            min_subgraph_size=0,
            precision="trt_fp16",
            dynamic=True,
            shape_range_file=file_path + "/shape_range.pbtxt",
            delete_pass_list=["trt_skip_layernorm_fuse_pass"],
        )

        del test_suite2  # destroy class to save memory


@pytest.mark.win
@pytest.mark.server
@pytest.mark.trt_fp16_multi_thread_bz1_precision
def test_trt_fp16_bz1_multi_thread():
    """
    compared trt fp16 batch_size=1 TNT_small multi_thread outputs with true val
    """
    check_model_exist()

    file_path = "./TNT_small"
    images_size = 224
    batch_size = 1
    test_suite = InferenceTest()
    test_suite.load_config(
        model_file="./TNT_small/inference.pdmodel",
        params_file="./TNT_small/inference.pdiparams",
    )
    images_list, npy_list = test_suite.get_images_npy(file_path, images_size)
    fake_input = np.array(images_list[0:batch_size]).astype("float32")
    input_data_dict = {"x": fake_input}
    output_data_dict = test_suite.get_truth_val(input_data_dict, device="gpu")

    del test_suite  # destroy class to save memory

    # collect shape for trt
    test_suite_c = InferenceTest()
    test_suite_c.load_config(
        model_file=file_path + "/inference.pdmodel",
        params_file=file_path + "/inference.pdiparams",
    )
    test_suite_c.collect_shape_info(
        model_path=file_path,
        input_data_dict=input_data_dict,
        device="gpu",
    )
    del test_suite_c

    test_suite2 = InferenceTest()
    test_suite2.load_config(
        model_file="./TNT_small/inference.pdmodel",
        params_file="./TNT_small/inference.pdiparams",
    )

    test_suite2.trt_bz1_multi_thread_test(
        input_data_dict,
        output_data_dict,
        delta=1e-2,
        min_subgraph_size=0,
        precision="trt_fp16",
        dynamic=True,
        shape_range_file="./TNT_small/shape_range.pbtxt",
        delete_pass_list=["trt_skip_layernorm_fuse_pass"],
    )

    del test_suite2  # destroy class to save memory
