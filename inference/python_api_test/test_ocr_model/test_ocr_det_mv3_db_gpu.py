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
from test_case import InferenceTest, clip_model_extra_op

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
        clip_model_extra_op(path_prefix="./ocr_det_mv3_db/inference", output_model_path="./ocr_det_mv3_db/inference")


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


@pytest.mark.win
@pytest.mark.server
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


@pytest.mark.win
@pytest.mark.server
@pytest.mark.gpu_more
def test_gpu_more_bz():
    """
    compared trt fp32 batch_size=1,2 ocr_det_mv3_db outputs with true val
    """
    check_model_exist()

    file_path = "./ocr_det_mv3_db"
    images_size = 640
    batch_size_pool = [1, 5, 10]
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

        test_suite2.gpu_more_bz_test(input_data_dict, output_data_dict, repeat=1, delta=2e-5)

        del test_suite2  # destroy class to save memory


@pytest.mark.jetson
@pytest.mark.gpu_more
def test_jetson_gpu_more_bz():
    """
    compared trt fp32 more batch_size ocr_det_mv3_db outputs with true val
    """
    check_model_exist()

    file_path = "./ocr_det_mv3_db"
    images_size = 640
    batch_size_pool = [1]
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

        test_suite2.gpu_more_bz_test(input_data_dict, output_data_dict, repeat=1, delta=3e-4)

        del test_suite2  # destroy class to save memory
