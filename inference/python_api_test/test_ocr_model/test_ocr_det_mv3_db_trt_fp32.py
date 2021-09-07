# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test ocr_det_mv3_db model
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
    ocr_det_mv3_db_url = "https://paddle-qa.bj.bcebos.com/inference_model/2.1.1/ocr/ocr_det_mv3_db.tgz"
    if not os.path.exists("./ocr_det_mv3_db/inference.pdiparams"):
        wget.download(ocr_det_mv3_db_url, out="./")
        tar = tarfile.open("ocr_det_mv3_db.tgz")
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
    test_suite.load_config(
        model_file="./ocr_det_mv3_db/inference.pdmodel", params_file="./ocr_det_mv3_db/inference.pdiparams"
    )
    test_suite.config_test()


@pytest.mark.p0
@pytest.mark.config_disablegpu_memory
def test_disable_gpu():
    """
    test no gpu resources occupied after disable gpu
    """
    check_model_exist()
    test_suite = InferenceTest()
    test_suite.load_config(
        model_file="./ocr_det_mv3_db/inference.pdmodel", params_file="./ocr_det_mv3_db/inference.pdiparams"
    )
    batch_size = 1
    fake_input = np.random.randn(batch_size, 3, 224, 224).astype("float32")
    input_data_dict = {"x": fake_input}
    test_suite.disable_gpu_test(input_data_dict)


@pytest.mark.p1
@pytest.mark.jetson
@pytest.mark.trt_fp32_multi_thread_bz1_dynamic_precision
def test_trt_fp32_bz1_dynamic_multi_thread():
    """
    compared trt fp32 batch_size=1 ocr_det_mv3_db multi_thread outputs with true val
    """
    check_model_exist()

    file_path = "./ocr_det_mv3_db"
    images_size = 640
    batch_size = 1
    max_batch_size = 1
    names = [
        "x",
        "conv2d_92.tmp_0",
        "conv2d_91.tmp_0",
        "conv2d_59.tmp_0",
        "nearest_interp_v2_1.tmp_0",
        "nearest_interp_v2_2.tmp_0",
        "conv2d_124.tmp_0",
        "nearest_interp_v2_3.tmp_0",
        "nearest_interp_v2_4.tmp_0",
        "nearest_interp_v2_5.tmp_0",
        "elementwise_add_7",
        "nearest_interp_v2_0.tmp_0",
    ]
    min_input_shape = [
        [1, 3, 50, 50],
        [1, 120, 20, 20],
        [1, 24, 10, 10],
        [1, 96, 20, 20],
        [1, 256, 10, 10],
        [1, 256, 20, 20],
        [1, 256, 20, 20],
        [1, 64, 20, 20],
        [1, 64, 20, 20],
        [1, 64, 20, 20],
        [1, 56, 2, 2],
        [1, 256, 2, 2],
    ]

    max_input_shape = [
        [max_batch_size, 3, 2000, 2000],
        [max_batch_size, 120, 400, 400],
        [max_batch_size, 24, 200, 200],
        [max_batch_size, 96, 400, 400],
        [max_batch_size, 256, 200, 200],
        [max_batch_size, 256, 400, 400],
        [max_batch_size, 256, 400, 400],
        [max_batch_size, 64, 400, 400],
        [max_batch_size, 64, 400, 400],
        [max_batch_size, 64, 400, 400],
        [max_batch_size, 56, 400, 400],
        [max_batch_size, 256, 400, 400],
    ]

    opt_input_shape = [
        [1, 3, 640, 640],
        [1, 120, 160, 160],
        [1, 24, 80, 80],
        [1, 96, 160, 160],
        [1, 256, 80, 80],
        [1, 256, 160, 160],
        [1, 256, 160, 160],
        [1, 64, 160, 160],
        [1, 64, 160, 160],
        [1, 64, 160, 160],
        [1, 56, 40, 40],
        [1, 256, 40, 40],
    ]
    test_suite = InferenceTest()
    test_suite.load_config(
        model_file="./ocr_det_mv3_db/inference.pdmodel", params_file="./ocr_det_mv3_db/inference.pdiparams"
    )
    images_list, npy_list = test_suite.get_images_npy(file_path, images_size)
    fake_input = np.array(images_list[0:batch_size]).astype("float32")
    input_data_dict = {"x": fake_input}
    output_data_dict = test_suite.get_truth_val(input_data_dict, device="gpu")

    del test_suite  # destroy class to save memory

    test_suite2 = InferenceTest()
    test_suite2.load_config(
        model_file="./ocr_det_mv3_db/inference.pdmodel", params_file="./ocr_det_mv3_db/inference.pdiparams"
    )
    test_suite2.trt_dynamic_multi_thread_test(
        input_data_dict,
        output_data_dict,
        gpu_mem=5000,
        max_batch_size=max_batch_size,
        repeat=1,
        delta=2e-5,
        names=names,
        min_input_shape=min_input_shape,
        max_input_shape=max_input_shape,
        opt_input_shape=opt_input_shape,
        precision="trt_fp32"
    )

    del test_suite2  # destroy class to save memory


@pytest.mark.p1
@pytest.mark.jetson
@pytest.mark.trt_fp32_more_bz_dynamic_precision
def test_trtfp32_more_bz_dynamic_bz():
    """
    compared trt fp32 batch_size=1,2 ocr_det_mv3_db outputs with true val
    """
    check_model_exist()

    file_path = "./ocr_det_mv3_db"
    images_size = 640
    batch_size_pool = [1, 5]
    max_batch_size = 5
    names = [
        "x",
        "conv2d_92.tmp_0",
        "conv2d_91.tmp_0",
        "conv2d_59.tmp_0",
        "nearest_interp_v2_1.tmp_0",
        "nearest_interp_v2_2.tmp_0",
        "conv2d_124.tmp_0",
        "nearest_interp_v2_3.tmp_0",
        "nearest_interp_v2_4.tmp_0",
        "nearest_interp_v2_5.tmp_0",
        "elementwise_add_7",
        "nearest_interp_v2_0.tmp_0",
    ]
    min_input_shape = [
        [1, 3, 50, 50],
        [1, 120, 20, 20],
        [1, 24, 10, 10],
        [1, 96, 20, 20],
        [1, 256, 10, 10],
        [1, 256, 20, 20],
        [1, 256, 20, 20],
        [1, 64, 20, 20],
        [1, 64, 20, 20],
        [1, 64, 20, 20],
        [1, 56, 2, 2],
        [1, 256, 2, 2],
    ]

    max_input_shape = [
        [max_batch_size, 3, 2000, 2000],
        [max_batch_size, 120, 400, 400],
        [max_batch_size, 24, 200, 200],
        [max_batch_size, 96, 400, 400],
        [max_batch_size, 256, 200, 200],
        [max_batch_size, 256, 400, 400],
        [max_batch_size, 256, 400, 400],
        [max_batch_size, 64, 400, 400],
        [max_batch_size, 64, 400, 400],
        [max_batch_size, 64, 400, 400],
        [max_batch_size, 56, 400, 400],
        [max_batch_size, 256, 400, 400],
    ]

    opt_input_shape = [
        [1, 3, 640, 640],
        [1, 120, 160, 160],
        [1, 24, 80, 80],
        [1, 96, 160, 160],
        [1, 256, 80, 80],
        [1, 256, 160, 160],
        [1, 256, 160, 160],
        [1, 64, 160, 160],
        [1, 64, 160, 160],
        [1, 64, 160, 160],
        [1, 56, 40, 40],
        [1, 256, 40, 40],
    ]
    for batch_size in batch_size_pool:
        test_suite = InferenceTest()
        test_suite.load_config(
            model_file="./ocr_det_mv3_db/inference.pdmodel", params_file="./ocr_det_mv3_db/inference.pdiparams"
        )
        images_list, npy_list = test_suite.get_images_npy(file_path, images_size)
        fake_input = np.array(images_list[0:batch_size]).astype("float32")
        input_data_dict = {"x": fake_input}
        output_data_dict = test_suite.get_truth_val(input_data_dict, device="gpu")

        del test_suite  # destroy class to save memory

        test_suite2 = InferenceTest()
        test_suite2.load_config(
            model_file="./ocr_det_mv3_db/inference.pdmodel", params_file="./ocr_det_mv3_db/inference.pdiparams"
        )

        test_suite2.trt_more_bz_dynamic_test(
            input_data_dict,
            output_data_dict,
            max_batch_size=max_batch_size,
            repeat=1,
            delta=2e-5,
            names=names,
            min_input_shape=min_input_shape,
            max_input_shape=max_input_shape,
            opt_input_shape=opt_input_shape,
            precision="trt_fp32"
        )

        del test_suite2  # destroy class to save memory
